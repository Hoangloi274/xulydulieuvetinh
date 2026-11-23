"""Live lightning nowcasting: every 10 minutes, ingest the newest radar, satellite and lightning, run the model, write
the forecast.

    # live (radar pulled from the radar machine, lightning from the iWeather feed)
    uv run python -m lightning_nowcast.live --ckpt models/lightning_nowcast.pt --work-dir live \\
        --radar-dir live/radar --radar-remote radarop@10.152.57.99:/path/to/raw --sat-raw-dir live/himawari \\
        --lightning-feed data/lightning/feed
    # replay over archived data, through the same code (for shadow runs and tests)
    ... --replay 2026-06-01T06:00 2026-06-01T07:00 --radar-dir data/raw/radar --sat-raw-dir data/raw/himawari \\
        --lightning-csv data/lightning/strokes_2025-01_2026-09_300km.csv --no-sat-download

Anchor T is the 10-min slot of the newest radar volumes; the forecast covers windows (T+10, T+W]. The newest input
to arrive is the 120 km volume of T (scanned T+3:03 to T+8:46), so the forecast is issued as soon as it is in:
  T+4:00, T+7:30  prefetch: the 300 km volume of T (ends T+2:57), satellite slot T-10 (complete on S3 by T+2:30-3:50),
                  lightning frames ending up to T
  T+8:40 ...      poll for the 120 km volume every --poll-sec; when it is in, decode it while the two newest
                  lightning frames (strokes to T+5) are fetched in one round trip, then run the model (~0.4 s on CPU)
  T+12:00         deadline: issue without the volume (the forecast's `health` says so)
A volume with fewer sweeps than the task has is taken as still being written and fetched again, until it is 2 min
past its nominal end. Strokes arrive with a delay (median 4 min, 90% within 6 min), so at ~T+9:15 about 90% of
the strokes in (T, T+5] are in.

Inputs reach the model through `LiveArchive`, which answers the same calls as the packed training stores
(`dataset.FrameSource`), so `dataset.Samples` and `decode_batch` build live inputs with the training code itself:
frames are encoded with the same codes (`pack.encode`), on the same grids and slot clock. A source missing for some
frames is zero-filled ("missing"), as the model saw in training through modality dropout, and reported in the
forecast's `health`.

Output per issue: <work-dir>/forecasts/YYYYmmdd/HHMM/{forecast.json, prob.npz, observed.npz} and <work-dir>/latest.json
(lightning_nowcast.product); the operator app (lightning_nowcast.app) reads them.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import subprocess
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NewType

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

from . import himawari_download, himawari_zarr, lightning_feed, regions
from .dataset import DataConfig, Samples, decode_batch
from .grid import (
    CENTER_LAT,
    CENTER_LON,
    INNER,
    OFFSET,
    OUTER,
    R120_VARS,
    R300_VARS,
    RES_KM,
    SAT_BANDS,
    Source,
    grid_lonlat,
    range_km_map,
)
from .model import load_model
from .pack import (
    CODECS,
    SAT_BAD_MIN,
    SAT_MAX_BAD_FRAC,
    codec_of,
    decode,
    encode,
    load_strokes,
    rasterize_strokes,
)
from .product import (
    LISTED_FROM,
    AreaWarning,
    Forecast,
    Level,
    Levels,
    Observed,
    SourceHealth,
    Window,
)
from .radar import GridSpec, volume_features
from .radar_zarr import parse_time_from_name, task_of_name
from .regions import DEFAULT_BOUNDARIES

Slot = NewType("Slot", int)    # 10-min slots since EPOCH: slot s covers the volumes/images nominally at its time
Slot5 = NewType("Slot5", int)  # 5-min stroke frames since EPOCH: frame k holds strokes in (t_k - 5 min, t_k]
EPOCH = datetime(2000, 1, 1)
SLOT = timedelta(minutes=10)
KEEP = 24  # slots kept in memory (4 h): history 6-7 slots, plus slack for late data and replay steps
PREFETCH_AT = (timedelta(minutes=4), timedelta(minutes=7, seconds=30))  # after T
RADAR_FROM = timedelta(minutes=8, seconds=40)  # start polling for the 120 km volume of T (its last sweep ends ~T+8:46)
RADAR_DEADLINE = timedelta(minutes=12)
REPLAY_ISSUE = timedelta(minutes=9, seconds=15)  # nominal issue time recorded by replays
# task -> (sweeps, scan duration from the start time in the file name)
VOLUMES = {Source.RADAR120: (8, timedelta(minutes=5, seconds=45)), Source.RADAR300: (4, timedelta(minutes=2, seconds=55))}
log = logging.getLogger("live")


def slot_of(t: datetime) -> Slot:
    """Slot whose time is nearest to t (radar files start ~0-3 min after their slot, like pack.store_index)."""
    return Slot(round((t - EPOCH) / SLOT))


def slot_time(s: Slot) -> datetime:
    return EPOCH + s * SLOT


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ------------------------------------------------------------------------------------------------ in-memory archive
FRAMES10 = (Source.RADAR120, Source.RADAR300, Source.SATELLITE)


class LiveArchive:
    """The recent frames of every source, encoded as in the packed stores (a dataset.FrameSource)."""

    def __init__(self) -> None:
        self.frames10: dict[Source, dict[Slot, np.ndarray]] = {k: {} for k in FRAMES10}
        self.sat_ok: dict[Slot, bool] = {}
        self.ltg: dict[Slot5, np.ndarray] = {}
        self._channels = {Source.RADAR120: len(R120_VARS), Source.RADAR300: len(R300_VARS), Source.SATELLITE: len(SAT_BANDS)}

    # FrameSource ------------------------------------------------------------------------------------
    def slot_time(self, s: int) -> np.datetime64:
        return np.datetime64(slot_time(Slot(int(s))), "m")

    def frames(self, key: Source, first: int, n: int, ys: slice, xs: slice) -> np.ndarray:
        out = np.zeros((n, self._channels[key], ys.stop - ys.start, xs.stop - xs.start), np.uint8)
        for j in range(n):
            f = self.frames10[key].get(Slot(first + j))
            if f is not None and (key != Source.SATELLITE or self.sat_ok.get(Slot(first + j), False)):
                out[j] = f[:, ys, xs]
        return out

    def ltg5(self, first5: int, n: int, ys: slice, xs: slice) -> np.ndarray:
        out = np.zeros((n, 2, ys.stop - ys.start, xs.stop - xs.start), np.uint8)
        for j in range(n):
            f = self.ltg.get(Slot5(first5 + j))
            if f is not None:
                out[j] = f[:, ys, xs]
        return out

    # bookkeeping ------------------------------------------------------------------------------------
    def put(self, key: Source, s: Slot, frame: np.ndarray, ok: bool = True) -> None:
        self.frames10[key][s] = frame
        if key == Source.SATELLITE:
            self.sat_ok[s] = ok

    def has(self, key: Source, s: Slot) -> bool:
        return s in self.frames10[key]

    def prune(self, newest: Slot) -> None:
        for buf in self.frames10.values():
            for s in [s for s in buf if s < newest - KEEP]:
                del buf[s]
        for k in [k for k in self.ltg if k < 2 * (newest - KEEP)]:
            del self.ltg[k]

    def health(self, anchor: Slot, cfg: DataConfig) -> dict[Source, SourceHealth]:
        t = cfg.t_in
        last = anchor - cfg.sat_lag_slots
        end5 = 2 * anchor + cfg.ltg_latency_min // 5
        h = {Source.RADAR120: SourceHealth(sum(self.has(Source.RADAR120, Slot(anchor - k)) for k in range(t)), t),
             Source.RADAR300: SourceHealth(sum(self.has(Source.RADAR300, Slot(anchor - k)) for k in range(t)), t),
             Source.SATELLITE: SourceHealth(sum(self.sat_ok.get(Slot(last - k), False) for k in range(t + 1)), t + 1),
             Source.LIGHTNING: SourceHealth(sum(Slot5(end5 - k) in self.ltg for k in range(2 * t)), 2 * t)}
        return {k: v for k, v in h.items() if k in cfg.inputs}


# ------------------------------------------------------------------------------------------------ sources
def source_of(path: Path) -> Source | None:
    task = task_of_name(path)
    return None if task is None else Source.RADAR120 if task == 120 else Source.RADAR300


def radar_frame(path: Path, now: datetime | None = None) -> tuple[Source, Slot, np.ndarray] | None:
    """(source, slot, codes) of one raw volume, encoded like pack.radar_reader; None if not a usable volume.
    Live (`now` given), a volume short of sweeps is rejected until 2 min after its nominal end: it may be partial."""
    key = source_of(path)
    t = parse_time_from_name(path)
    if key is None or t is None:
        return None
    grid = GridSpec(nx=INNER, ny=INNER, max_range_km=120.0) if key == Source.RADAR120 else GridSpec(nx=OUTER, ny=OUTER, max_range_km=300.0)
    sweeps, duration = VOLUMES[key]
    fresh = now is not None and now < t + duration + timedelta(minutes=2)
    try:
        f = volume_features(path, grid, min_sweeps=sweeps if fresh else 0)
    except Exception as exc:  # noqa: BLE001 - a corrupt volume is skipped, not fatal
        if fresh:  # most likely still being written or copied
            log.info("radar %s incomplete, retrying: %s", path.name, exc)
        else:
            log.warning("radar %s unreadable: %s", path.name, exc)
        return None
    if f is None:
        return None
    variables = R120_VARS if key == Source.RADAR120 else R300_VARS
    return key, slot_of(t), np.stack([encode(f[v], *codec_of(v)) for v in variables])


def slot_pattern(s: Slot) -> str:
    """File names of the volumes of slot s (both tasks start within its first 10 minutes)."""
    t = slot_time(s)
    return f"NHB{t:%y%m%d%H}{t.minute // 10}*.RAW*"


@dataclass(frozen=True)
class RsyncSource:
    """Raw volumes pulled from the radar machine (passwordless SSH). One SSH connection is kept open between pulls
    (ControlPersist), so a poll costs one round trip. Files are compared by size and time, so a volume copied while
    it was still being written is fetched again."""
    host: str  # user@host
    path: str

    @classmethod
    def parse(cls, spec: str) -> RsyncSource:
        host, path = spec.split(":", 1)
        return cls(host, path)

    def pull(self, into: Path, slots: Iterable[Slot]) -> None:
        (Path.home() / ".ssh").mkdir(mode=0o700, exist_ok=True)
        into.mkdir(parents=True, exist_ok=True)
        ssh = f"ssh -o BatchMode=yes -o ConnectTimeout=5 -o ControlMaster=auto -o ControlPersist=15m -o ControlPath={Path.home()}/.ssh/nowcast-%C"
        cmd = ["rsync", "-a", "-e", ssh, "--prune-empty-dirs", "--include=*/", *(f"--include={slot_pattern(s)}" for s in slots),
               "--exclude=*", f"{self.host}:{self.path}/", f"{into}/"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if r.returncode:
            log.warning("rsync failed (%d): %s", r.returncode, r.stderr.strip()[-300:])


class RadarIngest:
    """Converts raw volumes not yet seen, newest slots only, into the archive."""

    def __init__(self, raw_dir: Path, workers: int = 4) -> None:
        self.raw_dir = raw_dir
        self.seen: set[str] = set()
        self.pool = ThreadPoolExecutor(workers)  # xradar decoding releases the GIL for most of its work

    def files(self, s: Slot) -> list[Path]:
        return [p for p in self.raw_dir.rglob(slot_pattern(s)) if p.name not in self.seen]

    def arrived(self, key: Source, s: Slot) -> bool:
        """A file of the volume is on disk (not necessarily complete or read yet)."""
        return any(source_of(p) == key for p in self.files(s))

    def ingest(self, archive: LiveArchive, first: Slot, last: Slot, now: datetime | None = None) -> int:
        """Read the new volumes of slots first..last that the archive lacks. `now`: live wall clock (None: replay)."""
        slots = [Slot(s) for s in range(first, last + 1) if not all(archive.has(k, Slot(s)) for k in VOLUMES)]
        new = [p for s in slots for p in self.files(s)]
        n = 0
        clock = now or slot_time(last) + SLOT
        for p, res in zip(new, self.pool.map(lambda p: radar_frame(p, now), new)):
            t = parse_time_from_name(p)
            if res is not None or (t is not None and clock - t > timedelta(minutes=30)):
                self.seen.add(p.name)  # an unreadable or partial file is retried for 30 min
            if res is not None and first <= res[1] <= last and not archive.has(res[0], res[1]):
                archive.put(*res)
                n += 1
        return n


class SatelliteIngest:
    """Himawari slots: local raw folders (YYYY/MM/DD/HHMM, as himawari_download writes), downloaded from S3 when
    missing, resampled and encoded exactly as the training stores (nearest, aeqd, K x 100 then BT codes)."""

    def __init__(self, raw_dir: Path, cache_dir: Path, download: bool) -> None:
        self.raw_dir, self.cache_dir, self.download = raw_dir, cache_dir, download
        self.grid = himawari_zarr.GridSpec(CENTER_LAT, CENTER_LON, INNER, INNER, int(RES_KM * 1000))
        self._s3 = None

    def _fetch(self, t: datetime) -> bool:
        if self._s3 is None:
            self._s3 = himawari_download.make_s3_client(16)
        source, keys = himawari_download.choose_source_for_timestamp(self._s3, t)
        if source is None:
            return False
        status, msg = himawari_download.process_timestamp(self._s3, self.raw_dir, t, source, keys, False, 16, 1)
        log.info("satellite %s", msg)
        return status == "ok"

    def frame(self, t: datetime) -> tuple[np.ndarray, bool] | None:
        d = self.raw_dir / f"{t:%Y/%m/%d/%H%M}"
        if not (d / "manifest.json").exists() and not (self.download and self._fetch(t)):
            return None
        cfg = himawari_zarr.WorkerConfig(SAT_BANDS, self.grid, str(self.cache_dir), False)
        try:
            out = himawari_zarr.process_timestamp(str(d), cfg)
        except Exception as exc:  # noqa: BLE001 - a bad slot is skipped (the model saw missing satellite in training)
            log.warning("satellite %s unusable: %s", t, exc)
            return None
        bands, bad_frac = [], 0.0
        for b in SAT_BANDS:
            raw = out["arrays"][b].astype(np.float32)
            bad = raw >= SAT_BAD_MIN
            bad_frac = max(bad_frac, float(bad.mean()))
            bands.append(encode(np.where(bad, np.nan, raw / 100.0), *CODECS["bt"]))
        return np.ascontiguousarray(np.stack(bands)[:, ::-1]), bad_frac <= SAT_MAX_BAD_FRAC  # rows north->south -> south->north

    def ingest(self, archive: LiveArchive, slots: Iterable[Slot], until: datetime | None = None) -> int:
        """Newest slots first; stops at `until` (wall clock), leaving older slots for later (a cold start downloads
        ~7 slots at ~20 s each)."""
        n = 0
        for s in sorted(slots, reverse=True):
            if until is not None and utcnow() >= until:
                break
            if archive.has(Source.SATELLITE, s):
                continue
            res = self.frame(slot_time(s))
            if res is not None:
                archive.put(Source.SATELLITE, s, *res)
                n += 1
        return n


@dataclass(frozen=True)
class LightningFromFeed:
    root: Path  # lightning_feed archive (raw 5-min frames)


@dataclass(frozen=True)
class LightningFromCsv:
    path: Path  # stroke CSV in the training layout (replay)


LightningSource = LightningFromFeed | LightningFromCsv


class LightningIngest:
    def __init__(self, source: LightningSource) -> None:
        self.source = source
        self.client: lightning_feed.Client | None = None
        self.csv = load_strokes(source.path) if isinstance(source, LightningFromCsv) else None

    def refresh(self, first5: Slot5, last5: Slot5) -> None:
        """Live: fetch the feed's frames first5..last5 that are not final yet, concurrently (one round trip)."""
        match self.source:
            case LightningFromFeed(root):
                self.client = self.client or lightning_feed.Client()
                t0, t1 = EPOCH + (first5 - 1) * SLOT / 2, EPOCH + last5 * SLOT / 2  # feed frames are named by their start
                try:
                    lightning_feed.sync(self.client, root, t0, t1, settle_min=20.0, workers=4, log=lambda m: log.debug("lightning %s", m))
                except Exception as exc:  # noqa: BLE001 - feed outage: forecast from what is archived
                    log.warning("lightning feed: %s", exc)
            case LightningFromCsv():
                pass

    def strokes(self, t0: datetime, t1: datetime) -> pd.DataFrame:
        """Strokes with t0 < time <= t1 (the 5-min frames' convention)."""
        df = self._load(t0, t1)
        return df[(df["t"] > np.datetime64(t0)) & (df["t"] <= np.datetime64(t1))]

    def available(self, k: Slot5) -> bool:
        """Whether the strokes of frame k could be read (feed: a response is archived for the frame's period)."""
        match self.source:
            case LightningFromFeed(root):
                t = EPOCH + (k - 1) * SLOT / 2
                return lightning_feed.frame_path(root, t).exists() or lightning_feed.frame_path(root, t, partial=True).exists()
            case LightningFromCsv():
                return True

    def _load(self, t0: datetime, t1: datetime) -> pd.DataFrame:
        match self.source:
            case LightningFromFeed(root):
                s = lightning_feed.load_strokes(root, t0, t1 + timedelta(seconds=1))
                return pd.DataFrame({"t": s.time, "lat": s.lat, "lon": s.lon, "cloud": s.in_cloud.astype(int)})
            case LightningFromCsv():
                assert self.csv is not None
                return self.csv

    def ingest(self, archive: LiveArchive, first5: Slot5, last5: Slot5) -> None:
        """Rasterize the 5-min frames first5..last5 (re-done each cycle: frames keep filling until they settle)."""
        t_first = EPOCH + first5 * SLOT / 2
        t_last = EPOCH + last5 * SLOT / 2
        strokes = self.strokes(t_first - SLOT / 2, t_last)
        frames = rasterize_strokes(strokes, np.array([EPOCH + k * SLOT / 2 for k in range(first5, last5 + 1)], dtype="datetime64[s]"))
        for j, k in enumerate(range(first5, last5 + 1)):
            if self.available(Slot5(k)):
                archive.ltg[Slot5(k)] = frames[j]
            else:
                archive.ltg.pop(Slot5(k), None)


# ------------------------------------------------------------------------------------------------ forecaster
class Forecaster:
    def __init__(self, ckpt_path: Path, device: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = DataConfig.from_checkpoint(ckpt)
        self.model = load_model(ckpt, device)
        cal = ckpt.get("calibration")
        n = len(self.cfg.names)
        self.a = torch.tensor(cal["a"] if cal else [1.0] * n, device=device).view(1, -1, 1, 1)
        self.b = torch.tensor(cal["b"] if cal else [0.0] * n, device=device).view(1, -1, 1, 1)
        self.windows = tuple(Window(w, r) for w, r in zip(self.cfg.windows_min, self.cfg.radii_km))
        self.device = device
        self.name = f"{ckpt_path.stem}@{hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:8]}"
        self.predict(LiveArchive(), Slot(0))  # warm-up: the first call is several times slower

    @torch.inference_mode()
    def predict(self, archive: LiveArchive, anchor: Slot) -> np.ndarray:
        sample = Samples(archive, np.array([anchor]), self.cfg)[0]  # the training sample builder
        d = decode_batch(default_collate([sample]), self.cfg, self.device)
        logit, _ = self.model({k: d[k] for k in self.cfg.inputs}, d["static"])
        return torch.sigmoid(self.a * logit + self.b)[0].float().cpu().numpy()


class Areas:
    """The communes' cells on the 320 grid, within the scored domain (regions.commune_masks)."""

    def __init__(self, boundaries: Path, domain_km: float) -> None:
        lon, lat = grid_lonlat()
        units = regions.communes(boundaries)
        masks = regions.commune_masks(units, lon, lat, range_km_map() <= domain_km)
        self.communes = [(c, masks[c.code]) for c in units if c.code in masks]
        log.info("%d communes in the forecast domain (of %d)", len(self.communes), len(units))

    def warnings(self, prob: np.ndarray, levels: Levels, level_index: int) -> tuple[AreaWarning, ...]:
        """The communes reaching LISTED_FROM in some window, in bulletin order."""
        out = []
        for c, m in self.communes:
            p = tuple(round(float(v), 3) for v in prob[:, m].max(axis=1))
            if max(p) >= LISTED_FROM:
                out.append(AreaWarning(c.code, c.name, c.province, p, levels.of(p[level_index])))
        return tuple(out)


def summary(communes: tuple[AreaWarning, ...]) -> str:
    """'warning 12 (Tỉnh Đồng Tháp 7, Tỉnh Vĩnh Long 5), alert 4 (...)' for the log."""
    parts = []
    for level in (Level.WARNING, Level.ALERT):
        by = Counter(a.province for a in communes if a.level == level)
        top = ", ".join(f"{p} {n}" for p, n in by.most_common(3)) + (", ..." if len(by) > 3 else "")
        parts.append(f"{level.value} {sum(by.values())}" + (f" ({top})" if by else ""))
    return ", ".join(parts)


# ------------------------------------------------------------------------------------------------ the cycle
@dataclass
class Nowcaster:
    forecaster: Forecaster
    radar: RadarIngest
    satellite: SatelliteIngest
    lightning: LightningIngest
    areas: Areas
    levels: Levels
    work_dir: Path
    rsync: RsyncSource | None = None
    poll_sec: float = 2.0
    keep_raw: timedelta | None = timedelta(days=2)  # None: keep all raw inputs (archive for analysis)
    archive: LiveArchive = field(default_factory=LiveArchive)

    def __post_init__(self) -> None:
        self.level_index = next(i for i, x in enumerate(self.forecaster.windows)
                                if x.end_min == self.levels.window_min and x.radius_km == self.levels.radius_km)

    def history(self, anchor: Slot) -> tuple[Slot, Slot5]:
        """First radar slot and last lightning frame the model reads for `anchor`."""
        cfg = self.forecaster.cfg
        return Slot(anchor - cfg.t_in), Slot5(2 * anchor + cfg.ltg_latency_min // 5)

    def prefetch(self, anchor: Slot, live: bool) -> None:
        """What is available before the 120 km volume of `anchor` is complete."""
        cfg = self.forecaster.cfg
        first, end5 = self.history(anchor)
        if live and self.rsync:
            self.rsync.pull(self.radar.raw_dir, [Slot(s) for s in range(first, anchor + 1) if not all(self.archive.has(k, Slot(s)) for k in VOLUMES)])
        self.radar.ingest(self.archive, first, anchor, utcnow() if live else None)
        last = anchor - cfg.sat_lag_slots
        self.satellite.ingest(self.archive, [Slot(s) for s in range(last - cfg.t_in, last + 1)],
                              until=slot_time(anchor) + RADAR_FROM if live else None)
        if live:
            self.lightning.refresh(Slot5(end5 - 2 * cfg.t_in + 1), Slot5(end5 - 2))  # all but the two newest frames

    def wait_for_radar(self, anchor: Slot) -> None:
        """Live: poll until a file of the 120 km volume of `anchor` is on disk, or the deadline."""
        deadline = slot_time(anchor) + RADAR_DEADLINE
        while not self.radar.arrived(Source.RADAR120, anchor) and not self.archive.has(Source.RADAR120, anchor) and utcnow() < deadline:
            time.sleep(self.poll_sec)
            if self.rsync:
                self.rsync.pull(self.radar.raw_dir, [anchor])

    def issue(self, anchor: Slot, live: bool) -> Forecast:
        cfg = self.forecaster.cfg
        t0 = time.perf_counter()
        first, end5 = self.history(anchor)
        with ThreadPoolExecutor(1) as ex:
            # the newest strokes are fetched while the volume is decoded
            fetching = ex.submit(self.lightning.refresh, Slot5(end5 - 1), end5) if live else None
            self.radar.ingest(self.archive, first, anchor, utcnow() if live else None)
            deadline = slot_time(anchor) + RADAR_DEADLINE
            while live and not self.archive.has(Source.RADAR120, anchor) and utcnow() < deadline:  # partial file: fetch again
                time.sleep(self.poll_sec)
                if self.rsync:
                    self.rsync.pull(self.radar.raw_dir, [anchor])
                self.radar.ingest(self.archive, first, anchor, utcnow())
            t_radar = time.perf_counter()
            if fetching is not None:
                fetching.result()
        self.lightning.ingest(self.archive, Slot5(end5 - 2 * cfg.t_in + 1), end5)
        t_inputs = time.perf_counter()
        health = self.archive.health(anchor, cfg)
        prob = self.forecaster.predict(self.archive, anchor)
        areas, k = self.areas, self.level_index
        fc = Forecast(anchor=slot_time(anchor), issued=utcnow() if live else slot_time(anchor) + REPLAY_ISSUE,
                      model=self.forecaster.name, windows=self.forecaster.windows, levels=self.levels, prob=prob,
                      observed=self.observed(anchor, end5), health=health,
                      communes=areas.warnings(prob, self.levels, k))
        out = fc.write(self.work_dir)
        self.archive.prune(anchor)
        missing = [str(k) for k, h in health.items() if not h.complete]
        log.info("%s: issued T+%s (radar %.1f s, lightning +%.1f s, model and output %.1f s) -> %s | %s%s", fc.anchor,
                 str(fc.issued - fc.anchor)[2:7], t_radar - t0, t_inputs - t_radar, time.perf_counter() - t_inputs, out,
                 summary(fc.communes),
                 f" | incomplete inputs: {missing}" if missing else "")
        return fc

    def observed(self, anchor: Slot, end5: Slot5) -> Observed:
        cmax = np.full((OUTER, OUTER), np.nan, np.float32)
        for key, variables, at in ((Source.RADAR300, R300_VARS, slice(0, OUTER)), (Source.RADAR120, R120_VARS, slice(OFFSET, OFFSET + INNER))):
            frame = self.archive.frames10[key].get(anchor)
            if frame is not None:
                v = decode(frame[variables.index("CMAX")], *codec_of("CMAX"))
                cmax[at, at] = np.where(np.isnan(v), cmax[at, at], v)
        t1 = EPOCH + end5 * SLOT / 2
        df = self.lightning.strokes(t1 - timedelta(hours=1), t1)
        minutes = (df["t"].to_numpy().astype("datetime64[s]") - np.datetime64(slot_time(anchor), "s")) / np.timedelta64(60, "s")
        return Observed(cmax.astype(np.float16), minutes.astype(np.float32), df["lat"].to_numpy(np.float32),
                        df["lon"].to_numpy(np.float32), df["cloud"].to_numpy() == 0)

    def prune_raw(self, now: datetime, keep: timedelta = timedelta(days=2)) -> None:
        """Delete raw radar volumes and satellite slots older than `keep` (live only: the folders would grow forever)."""
        old = now - keep
        for p in self.radar.raw_dir.rglob("NHB*.RAW*"):
            t = parse_time_from_name(p)
            if t is not None and t < old:
                p.unlink(missing_ok=True)
        for day in self.satellite.raw_dir.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]"):
            if datetime.strptime("/".join(day.parts[-3:]), "%Y/%m/%d") + timedelta(days=1) < old:
                for f in sorted(day.rglob("*"), reverse=True):
                    f.unlink() if f.is_file() else f.rmdir()
                day.rmdir()

    def run_live(self) -> None:
        done = Slot(0)
        while True:
            now = utcnow()
            if self.keep_raw is not None and now.minute < 10 and now.hour == 0:  # once a day
                self.prune_raw(now, self.keep_raw)
            # the next anchor whose deadline has not passed
            anchor = Slot(max(done + 1, math.floor((now - RADAR_DEADLINE - EPOCH) / SLOT) + 1))
            steps = [(at, self.prefetch) for at in PREFETCH_AT] + [(RADAR_FROM, lambda a, live: self.wait_for_radar(a)), (RADAR_FROM, self.issue)]
            for at, step in steps:
                wait = (slot_time(anchor) + at - utcnow()).total_seconds()
                if wait > 0:
                    time.sleep(wait)
                try:
                    step(anchor, True)
                except Exception:  # noqa: BLE001 - one bad cycle must not stop the service
                    log.exception("cycle step for %s failed", slot_time(anchor))
            done = anchor

    def run_replay(self, first: datetime, last: datetime) -> None:
        s = slot_of(first)
        while s <= slot_of(last):
            self.prefetch(s, False)
            self.issue(s, False)
            s = Slot(s + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--radar-dir", type=Path, required=True, help="Raw radar volumes (NHB*.RAW*)")
    ap.add_argument("--radar-remote", type=RsyncSource.parse, default=None, help="user@host:/path to pull volumes from (live)")
    ap.add_argument("--sat-raw-dir", type=Path, required=True, help="Himawari HSD slots (YYYY/MM/DD/HHMM)")
    ap.add_argument("--no-sat-download", action="store_true", help="Use only slots already in --sat-raw-dir")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lightning-feed", type=Path, help="lightning_feed archive (live; IWEATHER_USER / IWEATHER_PASS)")
    src.add_argument("--lightning-csv", type=Path, help="Stroke CSV (replay)")
    ap.add_argument("--boundaries-dir", type=Path, default=DEFAULT_BOUNDARIES)
    ap.add_argument("--alert", type=float, default=Levels.alert)
    ap.add_argument("--warning", type=float, default=Levels.warning)
    ap.add_argument("--keep-raw", action="store_true", help="Live: never delete raw radar / satellite inputs (default: 2 days)")
    ap.add_argument("--poll-sec", type=float, default=2.0, help="Live: seconds between polls for the 120 km volume")
    ap.add_argument("--replay", nargs=2, type=datetime.fromisoformat, default=None, metavar=("FIRST", "LAST"), help="Anchors (UTC)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("satpy", "pyresample", "pyspectral"):  # one line per decoded file otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    forecaster = Forecaster(args.ckpt, args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lightning: LightningSource = LightningFromFeed(args.lightning_feed) if args.lightning_feed else LightningFromCsv(args.lightning_csv)
    nc = Nowcaster(
        forecaster=forecaster,
        radar=RadarIngest(args.radar_dir),
        satellite=SatelliteIngest(args.sat_raw_dir, args.work_dir / "resample_cache", download=not args.no_sat_download),
        lightning=LightningIngest(lightning),
        areas=Areas(args.boundaries_dir, forecaster.cfg.domain_km),
        levels=Levels(alert=args.alert, warning=args.warning),
        work_dir=args.work_dir,
        rsync=args.radar_remote,
        poll_sec=args.poll_sec,
        keep_raw=None if args.keep_raw else timedelta(days=2),
    )
    log.info("model %s on %s; inputs %s; outputs %s", forecaster.name, args.device, [str(s) for s in forecaster.cfg.inputs], forecaster.cfg.names)
    if args.replay:
        nc.run_replay(*args.replay)
    else:
        nc.run_live()


if __name__ == "__main__":
    main()
