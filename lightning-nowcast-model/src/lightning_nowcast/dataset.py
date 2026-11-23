"""Model samples from the packed monthly stores (lightning_nowcast.pack), and their decoding into tensors.

A sample is an anchor slot T (the 10-min slot of the newest radar volumes). DataLoader workers only read and crop the
uint8 codes; `decode_batch` turns them into model inputs and targets on the GPU. Time layout (minutes after T; the
forecast is issued at T+10, when both radar volumes are complete):

  r120, r300   slots T-50 ... T                     (+ T+10 ... T+60 of r120 Z for the auxiliary reflectivity head)
  sat          slots T-60 ... T-10 (sat_lag_slots)  (+ the slot before, for the B13 tendency)
  lightning    10-min sums ending T+5-50 ... T+5    (strokes reach the live feed within ~5 min: ltg_latency_min)
  targets      5-min stroke frames ending T+15 ... T+max(windows): window (T+10, T+W] within R km
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numcodecs
import numpy as np
import torch
import torch.nn.functional as F
import zarr

from .grid import (
    INNER,
    OFFSET,
    OUTER,
    R120_VARS,
    R300_VARS,
    RES_KM,
    SAT_BANDS,
    Source,
    range_km_map,
)
from .pack import CODECS

# input scaling to ~[0, 1]: (scale, log)
SCALING = {"Z": (60.0, False), "CMAX": (65.0, False), "Z_M10": (65.0, False), "Z_M20": (65.0, False),
           "ET30": (15.0, False), "VIL": (80.0, True)}
LTG_SCALE = math.log1p(30.0)
CHANNELS_PER_FRAME = {Source.RADAR120: len(R120_VARS) + 1, Source.RADAR300: len(R300_VARS) + 1,
                      Source.SATELLITE: len(SAT_BANDS) + 2, Source.LIGHTNING: 2}


@dataclass(frozen=True)
class DataConfig:
    """What the model sees and predicts. Stored in checkpoints (as "packed_config")."""
    t_in: int = 6
    sat_lag_slots: int = 1
    ltg_latency_min: int = 5
    windows_min: tuple[int, ...] = (20, 40, 60, 90, 120)  # window (T+10, T+W]
    radii_km: tuple[float, ...] = (10.0, 10.0, 10.0, 25.0, 25.0)
    target_kinds: str = "all"  # "all" (in-cloud + ground) or "cg"
    domain_km: float = 290.0  # targets are scored where the R-km disk stays within this range (strokes kept to 300 km)
    inputs: tuple[Source, ...] = (Source.RADAR120, Source.RADAR300, Source.SATELLITE, Source.LIGHTNING)
    n_ref: int = 6

    @property
    def names(self) -> list[str]:
        """Output names, e.g. "60r10" = (T+10, T+60] within 10 km."""
        return [f"{w}r{r:g}" for w, r in zip(self.windows_min, self.radii_km)]

    @property
    def n_future5(self) -> int:
        return (max(self.windows_min) - 10) // 5

    @classmethod
    def from_checkpoint(cls, ckpt: dict[str, Any]) -> DataConfig:
        stored: dict[str, Any] = dict(ckpt["packed_config"])
        if any(n > 1 for n in stored.pop("min_strokes", ()) or ()) or "env" in stored.get("inputs", ()):
            raise ValueError("checkpoint uses a feature this package no longer has (see archive/development)")
        known = {f.name for f in dataclasses.fields(cls)}
        kw: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in stored.items() if k in known}
        kw["inputs"] = tuple(Source(s) for s in kw.get("inputs", cls.inputs))
        return cls(**kw)

    def channels(self) -> dict[Source, int]:
        return {k: self.t_in * CHANNELS_PER_FRAME[k] for k in self.inputs}


class FrameSource(Protocol):
    """What Samples reads: uint8 codes of 10-min frames and 5-min stroke frames on one global slot clock.
    Implemented by the packed training stores (PackedStores) and by the live daemon's memory (live.LiveArchive)."""

    def slot_time(self, s: int) -> np.datetime64 | np.ndarray: ...
    def frames(self, key: Source, first: int, n: int, ys: slice, xs: slice) -> np.ndarray: ...
    def ltg5(self, first5: int, n: int, ys: slice, xs: slice) -> np.ndarray: ...


class PackedStores:
    """Global 10-min slot index over the packed monthly stores; chunk files are read directly, per process."""

    def __init__(self, root: Path) -> None:
        self.paths = sorted(root.glob("????_??.zarr"))
        if not self.paths:
            raise FileNotFoundError(f"no packed stores in {root}")
        self.month0 = [np.datetime64(f"{p.stem[:4]}-{p.stem[5:7]}-01T00:00", "m") for p in self.paths]
        self.sizes = []
        ok: dict[Source, list[np.ndarray]] = {k: [] for k in (Source.RADAR120, Source.RADAR300, Source.SATELLITE)}
        for p in self.paths:
            g = zarr.open_group(str(p), mode="r")
            n = int(np.asarray(g["slot"]).size)
            self.sizes.append(n)
            for k, flags in ok.items():  # a source not packed (yet) in a store counts as missing
                flags.append(np.asarray(g[f"{k}_ok"], bool) if f"{k}_ok" in g else np.zeros(n, bool))
        self.start = self.month0[0]
        offsets = [int((m - self.start) / np.timedelta64(10, "m")) for m in self.month0]
        total = offsets[-1] + self.sizes[-1]
        self.ok = {k: np.zeros(total, bool) for k in ok}
        self.month_of = np.full(total, -1, np.int32)
        for i, (o, s) in enumerate(zip(offsets, self.sizes)):
            for k in ok:
                self.ok[k][o:o + s] = ok[k][i]
            self.month_of[o:o + s] = i
        self.offsets = np.array(offsets)
        self._pid = -1
        self._arrays: list[dict] = []
        # decoded-chunk cache, per process: consecutive anchors share most of their frames (see RunBatchSampler)
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.cache_bytes = 64 << 20
        self._cached = 0

    def __len__(self) -> int:
        return self.month_of.size

    def slot_time(self, s: np.ndarray | int) -> np.ndarray:
        return self.start + np.asarray(s) * np.timedelta64(10, "m")

    def slot_of(self, t: np.datetime64) -> int:
        return int((np.datetime64(t, "m") - self.start) / np.timedelta64(10, "m"))

    def _open(self) -> None:
        if self._pid != os.getpid():  # per process: forked workers start with an empty cache
            self._arrays = []
            for p in self.paths:
                meta = {}
                for k in ("r120", "r300", "sat", "ltg"):
                    if not (p / k / ".zarray").exists():
                        continue
                    z = json.loads((p / k / ".zarray").read_text())
                    if z["zarr_format"] != 2 or z.get("filters") or z["order"] != "C" or z["chunks"][1:] != z["shape"][1:]:
                        raise ValueError(f"{p / k}: unexpected layout for direct chunk reads")
                    meta[k] = {"dir": p / k, "codec": numcodecs.get_codec(z["compressor"]), "dtype": np.dtype(z["dtype"]),
                               "chunk": tuple(z["chunks"]), "n": z["shape"][0], "sep": z.get("dimension_separator", ".")}
                self._arrays.append(meta)
            self._pid = os.getpid()
            self._cache.clear()
            self._cached = 0

    def _read_chunk(self, meta: dict, i: int) -> np.ndarray:
        """Chunk i read straight from its file (zarr v2: one compressed blob per chunk; zarr's own read path costs
        ~3.5 ms of event-loop overhead per call on a busy server, more than decompression). A missing file is an
        all-zero chunk (zarr does not write chunks equal to the fill value)."""
        shape = list(meta["chunk"])
        shape[0] = min(shape[0], meta["n"] - i * shape[0])
        name = meta["sep"].join([str(i)] + ["0"] * (len(shape) - 1))
        try:
            raw = (meta["dir"] / name).read_bytes()
        except FileNotFoundError:
            return np.zeros(shape, meta["dtype"])
        a = np.frombuffer(meta["codec"].decode(raw), dtype=meta["dtype"])
        return a.reshape([meta["chunk"][0]] + shape[1:])[: shape[0]]

    def _chunk(self, key: str, m: int, i: int) -> np.ndarray:
        """Chunk i of array `key` of month m, decoded (cached): one frame of r120/r300/sat, one hour of ltg."""
        k = (key, m, i)
        a = self._cache.get(k)
        if a is not None:
            self._cache.move_to_end(k)
            return a
        a = self._read_chunk(self._arrays[m][key], i)
        if key != "ltg":
            a = a[0]
        self._cache[k] = a
        self._cached += a.nbytes
        while self._cached > self.cache_bytes:
            _, old = self._cache.popitem(last=False)
            self._cached -= old.nbytes
        return a

    def frames(self, key: Source, first: int, n: int, ys: slice, xs: slice) -> np.ndarray:
        """Frames first ... first + n - 1 (global slots) of a 10-min source, (n, C, h, w); zeros (= missing) outside
        the stores."""
        self._open()
        c = {Source.RADAR120: len(R120_VARS), Source.RADAR300: len(R300_VARS), Source.SATELLITE: len(SAT_BANDS)}[key]
        out = np.zeros((n, c, ys.stop - ys.start, xs.stop - xs.start), np.uint8)
        for j, s in enumerate(range(first, first + n)):
            m = self.month_of[s] if 0 <= s < len(self) else -1
            if m >= 0 and key in self._arrays[m]:  # a source absent from a store reads as missing (zeros)
                out[j] = self._chunk(key, m, s - self.offsets[m])[:, ys, xs]
        return out

    def ltg5(self, first5: int, n: int, ys: slice, xs: slice) -> np.ndarray:
        """(n, 2, h, w) stroke counts of the 5-min frames first5 ... first5 + n - 1 (global 5-min index)."""
        self._open()
        out = np.zeros((n, 2, ys.stop - ys.start, xs.stop - xs.start), np.uint8)
        for j, k in enumerate(range(first5, first5 + n)):
            s = k // 2
            m = self.month_of[s] if 0 <= s < len(self) else -1
            if m >= 0 and "ltg" in self._arrays[m]:
                local = k - 2 * self.offsets[m]
                out[j] = self._chunk("ltg", m, local // 12)[local % 12, :, ys, xs]
        return out


def _paste(src: np.ndarray, y0: int, x0: int, size: int) -> np.ndarray:
    """Crop (y0, x0, size) of the OUTER grid taken from an INNER-grid array (..., 256, 256) placed at OFFSET."""
    out = np.zeros(src.shape[:-2] + (size, size), src.dtype)
    sy0, sy1 = max(y0 - OFFSET, 0), min(y0 + size - OFFSET, INNER)
    sx0, sx1 = max(x0 - OFFSET, 0), min(x0 + size - OFFSET, INNER)
    if sy1 > sy0 and sx1 > sx0:
        out[..., sy0 + OFFSET - y0:sy1 + OFFSET - y0, sx0 + OFFSET - x0:sx1 + OFFSET - x0] = src[..., sy0:sy1, sx0:sx1]
    return out


@dataclass
class Samples(torch.utils.data.Dataset):
    archive: FrameSource
    anchors: np.ndarray  # global slot indices
    cfg: DataConfig
    crop: int | None = None
    _range: np.ndarray = field(default_factory=range_km_map, repr=False)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:  # ty: ignore[lost-keyword-for-regular-parameter]
        s = int(self.anchors[i])
        size = self.crop or OUTER
        if self.crop:
            y0, x0 = (int(v) for v in torch.randint(0, OUTER - size + 1, (2,)))
        else:
            y0 = x0 = 0
        ys, xs = slice(y0, y0 + size), slice(x0, x0 + size)
        cfg, a = self.cfg, self.archive
        h0 = s - cfg.t_in + 1
        out: dict[str, np.ndarray] = {"anchor": np.array(s, np.int64), "range_km": self._range[ys, xs]}
        # INNER-grid sources are read in full and pasted (their chunks are whole frames anyway);
        # the r120 history and the future reflectivity of the auxiliary head are one read
        full = slice(0, INNER)
        r120 = a.frames(Source.RADAR120, h0, cfg.t_in + cfg.n_ref, full, full)
        if Source.RADAR120 in cfg.inputs:
            out[Source.RADAR120] = _paste(r120[:cfg.t_in], y0, x0, size)
        out["ref"] = _paste(r120[cfg.t_in:, 0], y0, x0, size)
        if Source.RADAR300 in cfg.inputs:
            out[Source.RADAR300] = a.frames(Source.RADAR300, h0, cfg.t_in, ys, xs)
        if Source.SATELLITE in cfg.inputs:
            last = s - cfg.sat_lag_slots
            out[Source.SATELLITE] = _paste(a.frames(Source.SATELLITE, last - cfg.t_in, cfg.t_in + 1, full, full), y0, x0, size)
        # lightning: 5-min frames; global 5-min index of the frame ending at slot time T is 2 * s
        end_in = 2 * s + cfg.ltg_latency_min // 5  # frame ending at T + latency
        both = a.ltg5(end_in - 2 * cfg.t_in + 1, 2 * cfg.t_in, ys, xs)
        if Source.LIGHTNING in cfg.inputs:
            out[Source.LIGHTNING] = both
        fut = a.ltg5(2 * s + 3, cfg.n_future5, ys, xs)  # frames ending T+15 ... T+max(W)
        out["fut"] = fut[:, 1] if cfg.target_kinds == "cg" else np.minimum(fut[:, 0].astype(np.uint16) + fut[:, 1], 255).astype(np.uint8)
        now = both[-2:].sum(axis=0) if cfg.ltg_latency_min == 5 else a.ltg5(2 * s, 2, ys, xs).sum(axis=0)
        out["now"] = (now[1] if cfg.target_kinds == "cg" else now.sum(axis=0)).astype(np.uint8)  # (T-5, T+5]
        return out


def worker_context(num_workers: int) -> str | None:
    """DataLoader workers fork, as before Python 3.14, whose default (forkserver) re-imports torch and the stores
    in every worker: tens of seconds per worker start."""
    return "fork" if num_workers > 0 else None


class RunBatchSampler(torch.utils.data.Sampler):
    """Batches of `batch_size // run` runs of `run` consecutive anchors, the runs drawn at random each epoch.

    A DataLoader worker builds a whole batch, so within a run most frames come from its chunk cache (anchors 10 min
    apart share 11 of 12 radar frames): far less decompression on a CPU-bound server. Every anchor is still used
    once per epoch with its own random crop. `anchors` must be sorted (the dataset's order)."""

    def __init__(self, anchors: np.ndarray, batch_size: int, run: int = 4, drop_last: bool = True, seed: int = 0) -> None:
        self.batch_size, self.run, self.drop_last = batch_size, max(1, run), drop_last
        idx = np.arange(len(anchors))
        breaks = np.flatnonzero(np.diff(anchors) != 1) + 1  # gaps in the anchor sequence end a run
        self.runs = [r[i:i + self.run] for r in np.split(idx, breaks) for i in range(0, len(r), self.run)]
        self.epoch, self.seed = 0, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        flat = np.concatenate([self.runs[i] for i in rng.permutation(len(self.runs))])
        for b in range(0, len(flat), self.batch_size):
            batch = flat[b:b + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch.tolist()

    def __len__(self) -> int:
        n = sum(len(r) for r in self.runs)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


def valid_anchors(stores: PackedStores, cfg: DataConfig, start: np.datetime64 | None = None,
                  end: np.datetime64 | None = None) -> np.ndarray:
    """Slots with every history frame of every input present (lightning is always present)."""
    n = len(stores)
    s = np.arange(n)
    ok = np.ones(n, bool)
    for src in cfg.inputs:
        if src == Source.LIGHTNING:
            continue
        flag = stores.ok[src]
        lag = cfg.sat_lag_slots if src == Source.SATELLITE else 0
        k0 = -lag - (cfg.t_in if src == Source.SATELLITE else cfg.t_in - 1)  # sat also needs the slot before (tendency)
        for k in range(k0, -lag + 1):
            idx = s + k
            ok &= (idx >= 0) & flag[np.clip(idx, 0, n - 1)]
    ok &= s + max(cfg.windows_min) // 10 < n  # targets inside the stores
    if start is not None:
        ok &= stores.slot_time(s) >= np.datetime64(start, "m")
    if end is not None:
        ok &= stores.slot_time(s) <= np.datetime64(end, "m")
    return s[ok]


def split_anchors(stores: PackedStores, anchors: np.ndarray, purge_hours: float) -> tuple[np.ndarray, np.ndarray]:
    """Validation = ISO week % 5 == 0. Training anchors within purge_hours of any validation anchor are dropped, so
    history and target windows never overlap across the split."""
    t = stores.slot_time(anchors).astype("datetime64[D]")
    weeks = np.array([d.astype(object).isocalendar()[1] for d in t])
    is_val = weeks % 5 == 0
    val, train = anchors[is_val], anchors[~is_val]
    if val.size:
        gap = int(purge_hours * 6)
        pos = np.searchsorted(val, train)
        d_next = np.where(pos < val.size, val[np.minimum(pos, val.size - 1)] - train, 10**9)
        d_prev = np.where(pos > 0, train - val[np.maximum(pos - 1, 0)], 10**9)
        train = train[(d_next > gap) & (d_prev > gap)]
    return train, val


# ---------------------------------------------------------------------------------------------- GPU side
def _decode(code: torch.Tensor, lo: float, step: float) -> tuple[torch.Tensor, torch.Tensor]:
    valid = code > 0
    return torch.where(valid, (code.float() - 1.0) * step + lo, torch.zeros((), device=code.device)), valid


def _scale(var: str, v: torch.Tensor) -> torch.Tensor:
    scale, log = SCALING[var]
    v = v.clamp(0.0, scale)
    return torch.log1p(v) / math.log1p(scale) if log else v / scale


_LUTS: dict[tuple[str, str], torch.Tensor] = {}


def _lut(var: str, device: torch.device) -> torch.Tensor:
    """256-entry table: uint8 code -> scaled model input (0 for missing), the same values as _scale(_decode(code))."""
    key = (var, str(device))
    if key not in _LUTS:
        lo, step = CODECS["ET30"] if var == "ET30" else CODECS["VIL"] if var == "VIL" else CODECS["dbz"]
        v, _ = _decode(torch.arange(256, device=device, dtype=torch.uint8), lo, step)
        _LUTS[key] = _scale(var, v)
    return _LUTS[key]


def _radar(code: torch.Tensor, variables: tuple[str, ...]) -> torch.Tensor:
    """(B, T, C, H, W) codes -> (B, T, C + 1, H, W): scaled fields and the lowest-sweep validity."""
    chans = [_lut(var, code.device)[code[:, :, k].long()] for k, var in enumerate(variables)]
    chans.append((code[:, :, 0] > 0).float())
    return torch.stack(chans, dim=2)


def dilate(x: torch.Tensor, r_px: int) -> torch.Tensor:
    """Binary disk dilation of (B, 1, H, W): the disk is a stack of horizontal segments, so the result is the maximum
    over row offsets dy of a row-wise max-pool of half-width sqrt(r^2 - dy^2), shifted by dy."""
    x = x.float()
    if r_px <= 0:
        return x
    out = torch.zeros_like(x)
    pooled: dict[int, torch.Tensor] = {}
    pad = F.pad(x, (0, 0, r_px, r_px))  # rows padded so that shifted views stay in range
    for dy in range(-r_px, r_px + 1):
        w = math.isqrt(r_px * r_px - dy * dy)
        if w not in pooled:
            pooled[w] = F.max_pool2d(pad, kernel_size=(1, 2 * w + 1), stride=1, padding=(0, w))
        out = torch.maximum(out, pooled[w][:, :, r_px + dy:r_px + dy + x.shape[2]])
    return out


def decode_batch(b: dict[str, torch.Tensor], cfg: DataConfig, device: str,
                 drop: dict[Source, float] | None = None) -> dict[str, torch.Tensor]:
    """Model inputs (float, keyed by Source), targets y / mask m (B, n_windows, H, W), persistence, auxiliary
    reflectivity. `drop`: per-source probability of blanking a sample's input (modality dropout, training only)."""
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    out: dict[str, torch.Tensor] = {}
    if Source.RADAR120 in b:
        out[Source.RADAR120] = _radar(b[Source.RADAR120], R120_VARS).flatten(1, 2)
    if Source.RADAR300 in b:
        out[Source.RADAR300] = _radar(b[Source.RADAR300], R300_VARS).flatten(1, 2)
    if Source.SATELLITE in b:
        bt, valid = _decode(b[Source.SATELLITE], *CODECS["bt"])  # (B, T+1, 5, H, W)
        x = torch.where(valid, (bt - 200.0) / 80.0, torch.zeros((), device=bt.device))
        i13 = SAT_BANDS.index("B13")
        both = valid[:, 1:, i13] & valid[:, :-1, i13]
        tend = torch.where(both, (bt[:, 1:, i13] - bt[:, :-1, i13]) / 10.0, torch.zeros((), device=bt.device))
        out[Source.SATELLITE] = torch.cat([x[:, 1:], tend[:, :, None], valid[:, 1:, i13, None].float()], dim=2).flatten(1, 2)
    if Source.LIGHTNING in b:
        c = b[Source.LIGHTNING].float()  # (B, 2T, 2, H, W) 5-min frames, oldest first -> 10-min sums
        c = c.view(c.shape[0], -1, 2, *c.shape[2:]).sum(dim=2)
        out[Source.LIGHTNING] = (torch.log1p(c) / LTG_SCALE).flatten(1, 2)
    for k, p in (drop or {}).items():
        if k in out and p > 0:
            out[k][torch.rand(out[k].shape[0], device=device) < p] = 0.0
    rng = b["range_km"].float()
    out["static"] = (rng / 300.0)[:, None]
    fut = (b["fut"] > 0).float()  # (B, F, H, W): strokes in the 5-min frames ending T+15 ...
    now = (b["now"] > 0).float()[:, None]
    ys, ms, ps = [], [], []
    for w, r in zip(cfg.windows_min, cfg.radii_km):
        r_px = round(r / RES_KM)
        ys.append(dilate(fut[:, :(w - 10) // 5].amax(dim=1, keepdim=True), r_px))
        ms.append((rng <= cfg.domain_km - r).float()[:, None])
        ps.append(dilate(now, r_px))
    out["y"], out["m"], out["pers"] = torch.cat(ys, 1), torch.cat(ms, 1), torch.cat(ps, 1)
    ref, ref_ok = _decode(b["ref"], *CODECS["dbz"])
    out["y_ref"], out["ref_ok"] = _scale("Z", ref), ref_ok.float()
    return out
