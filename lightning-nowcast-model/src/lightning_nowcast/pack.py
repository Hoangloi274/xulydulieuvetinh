"""Pack the training inputs into compact monthly stores on one regular 10-min clock.

    uv run python -m lightning_nowcast.pack --out data/stores/packed --months 2025-01 2025-12 \
        --radar120 data/stores/radar --radar300 data/stores/radar --sat data/stores/satellite \
        --strokes-csv data/lightning/strokes_2025-01_2026-09_300km.csv

Training used to read one compressed .npz per sample (~3 MB, zip members) over NFS, at ~9 samples/s. These stores
hold every source once, as uint8 frames on a fixed slot grid, so a sample is a handful of whole-chunk reads and the
whole year fits on a small local disk. Each month <out>/YYYY_MM.zarr (zarr v2, written with xarray) holds:

  r120  (slot, 6, 256, 256)   Z, CMAX, Z_M10, Z_M20, ET30, VIL of the 120 km volume (8 sweeps); slot T = volume T+3
  r300  (slot, 5, 320, 320)   Z, CMAX, Z_M10, ET30, VIL of the 300 km long-range volume (4 sweeps); slot T = volume T+0
  sat   (slot, 5, 256, 256)   Himawari B08, B10, B13, B15, B16 brightness temperature, rows south -> north
  ltg   (slot5, 2, 320, 320)  strokes per pixel in (t - 5 min, t] on a 5-min clock: in-cloud, cloud-to-ground (capped 255)
  r120_ok, r300_ok, sat_ok    (slot,) the frame exists (sat: at most 1% bad pixels, as features.sat_slots_ok)

The 320 grid (2 km, 640 km wide) shares its cell centres with the 256 grid, which sits at offset 32. Codes: 0 =
missing / not sampled; reflectivity (v + 10 dBZ) * 2 + 1, ET30 km * 10 + 1, VIL kg m-2 / 0.4 + 1, brightness
temperature (K - 170) / 0.6 + 1 (see CODECS). Strokes are placed with the azimuthal equidistant projection about the
radar, the projection of the radar and satellite grids.

--sources picks what to pack (default: all). Sources already in a month's store are skipped, so a store can be
built in steps, e.g. r300 and ltg where the long-range stores and the stroke CSV are, then r120 and sat on the
training server (--force rebuilds the named sources).
"""
from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc

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
    aeqd_km,
)

# code = clip(round((v - lo) / step), 0, 254) + 1; 0 = missing
CODECS = {"dbz": (-10.0, 0.5), "ET30": (0.0, 0.1), "VIL": (0.0, 0.4), "bt": (170.0, 0.6)}
STEP = np.timedelta64(10, "m")
STEP5 = np.timedelta64(5, "m")
SAT_BAD_MIN = 65534  # K x 100: the satellite converter's missing value
SAT_MAX_BAD_FRAC = 0.01
BLOCK = 144  # slots per read/write block (one day)


def codec_of(var: str) -> tuple[float, float]:
    return CODECS["ET30"] if var == "ET30" else CODECS["VIL"] if var == "VIL" else CODECS["dbz"]


def encode(v: np.ndarray, lo: float, step: float) -> np.ndarray:
    ok = np.isfinite(v)
    code = np.clip(np.rint((np.where(ok, v, lo) - lo) / step), 0, 254).astype(np.uint8) + 1
    return np.where(ok, code, 0).astype(np.uint8)


def decode(code: np.ndarray, lo: float, step: float) -> np.ndarray:
    """float32 values, NaN where code == 0."""
    v = (code.astype(np.float32) - 1.0) * step + lo
    v[code == 0] = np.nan
    return v



def month_slots(year: int, month: int) -> np.ndarray:
    t0 = np.datetime64(f"{year:04d}-{month:02d}-01T00:00", "m")
    t1 = np.datetime64(f"{year + (month == 12):04d}-{month % 12 + 1:02d}-01T00:00", "m")
    return np.arange(t0, t1, STEP)


def store_index(paths: list[Path]) -> list[tuple[xr.Dataset, np.ndarray]]:
    """Open stores; per store the slot (10-min rounded time) of each frame."""
    out = []
    for p in paths:
        ds = xr.open_zarr(p, consolidated=False, chunks=None)
        t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
        out.append((ds, ((t + 300) // 600 * 600).astype("datetime64[s]").astype("datetime64[m]")))
    return out


def fill_source(stores, slots: np.ndarray, out: np.ndarray, ok: np.ndarray, read_frames) -> None:
    """For each slot, the first frame (in store order, earliest time) that rounds to it."""
    filled = np.zeros(slots.size, bool)
    for ds, st in stores:
        pos = np.searchsorted(slots, st)
        hit = (pos < slots.size) & (slots[np.minimum(pos, slots.size - 1)] == st)
        for b0 in range(0, slots.size, BLOCK):
            sel = np.flatnonzero(hit & (pos >= b0) & (pos < b0 + BLOCK))
            sel = sel[~filled[pos[sel]]]
            if sel.size == 0:
                continue
            _, first = np.unique(pos[sel], return_index=True)  # duplicates: keep the first (earliest)
            sel = sel[first]
            frames, good = read_frames(ds, sel)
            out[pos[sel]] = frames
            ok[pos[sel]] = good
            filled[pos[sel]] = True


def radar_reader(variables: tuple[str, ...]):
    def read(ds: xr.Dataset, idx: np.ndarray):
        lo, hi = int(idx.min()), int(idx.max()) + 1
        block = {v: ds[v][lo:hi].values for v in variables}
        frames = np.stack([np.stack([encode(block[v][i - lo], *codec_of(v)) for v in variables]) for i in idx])
        return frames, np.ones(idx.size, bool)
    return read


def sat_reader(ds: xr.Dataset, idx: np.ndarray):
    lo, hi = int(idx.min()), int(idx.max()) + 1
    north_first = ds.attrs.get("row_order", "north_to_south") == "north_to_south"
    block = {b: ds[b][lo:hi].values for b in SAT_BANDS}
    frames, good = [], []
    for i in idx:
        bands = []
        bad_frac = 0.0
        for b in SAT_BANDS:
            raw = block[b][i - lo].astype(np.float32)
            bad = ~np.isfinite(raw) | (raw >= SAT_BAD_MIN)
            bad_frac = max(bad_frac, float(bad.mean()))
            bands.append(encode(np.where(bad, np.nan, raw / 100.0), *CODECS["bt"]))
        x = np.stack(bands)
        frames.append(x[:, ::-1] if north_first else x)
        good.append(bad_frac <= SAT_MAX_BAD_FRAC)
    return np.ascontiguousarray(np.stack(frames)), np.array(good)


def rasterize_strokes(strokes: pd.DataFrame, slots5: np.ndarray) -> np.ndarray:
    """(slot5, 2, 320, 320) uint8 stroke counts in (t - 5 min, t]: channel 0 in-cloud, 1 cloud-to-ground."""
    out = np.zeros((slots5.size, 2, OUTER, OUTER), np.uint8)
    t = strokes["t"].to_numpy()
    k = np.searchsorted(slots5, t, side="left")  # first slot end >= t, i.e. t in (end - 5, end]
    x, y = aeqd_km(strokes["lat"].to_numpy(), strokes["lon"].to_numpy())
    ix = np.floor(x / RES_KM + OUTER / 2).astype(int)
    iy = np.floor(y / RES_KM + OUTER / 2).astype(int)
    keep = (k < slots5.size) & (ix >= 0) & (ix < OUTER) & (iy >= 0) & (iy < OUTER)
    kind = (strokes["cloud"].to_numpy() == 0).astype(int)  # UALF cloud indicator: 1 = IC, 0 = CG
    flat, count = np.unique(np.ravel_multi_index((k[keep], kind[keep], iy[keep], ix[keep]), out.shape), return_counts=True)
    out.ravel()[flat] = np.minimum(count, 255)
    return out


def load_strokes(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, usecols=["time_utc", "lat", "lon", "c22"], encoding="utf-8-sig")
    df = df.drop_duplicates(["time_utc", "lat", "lon"])
    t = pd.to_datetime(df["time_utc"]).to_numpy().astype("datetime64[s]")
    return pd.DataFrame({"t": t, "lat": df["lat"].to_numpy(), "lon": df["lon"].to_numpy(), "cloud": df["c22"].to_numpy()})


def pack_month(year: int, month: int, args: argparse.Namespace) -> str:
    out = args.out / f"{year:04d}_{month:02d}.zarr"
    have = set()
    if out.exists():
        have = set(xr.open_zarr(out, consolidated=False).data_vars)
    todo = [src for src in args.sources if args.force or src not in have]
    if not todo:
        return f"{year}-{month:02d}: {', '.join(args.sources)} already packed"
    t0 = time.time()
    slots = month_slots(year, month)
    n = slots.size
    slots5 = np.arange(slots[0], slots[-1] + STEP, STEP5).astype("datetime64[s]")

    def stores(roots: list[Path] | None, pattern: str) -> list[Path]:
        return [p for root in roots or [] for p in sorted(root.glob(pattern))]

    # stores of the neighbouring months may hold frames that round into this month (e.g. 23:58 -> 00:00)
    near = lambda roots, pattern: [p for p in stores(roots, pattern if "*" in pattern else f"*{pattern}")  # noqa: E731
                                   if any(s in p.name for s in neighbours(year, month))]
    data: dict[str, tuple] = {}
    notes = []
    if "r120" in todo:
        a = np.zeros((n, len(R120_VARS), INNER, INNER), np.uint8); ok = np.zeros(n, bool)
        fill_source(store_index(near(args.radar120, "_120km.zarr")), slots, a, ok, radar_reader(R120_VARS))
        data["r120"] = (("slot", "c120", "y256", "x256"), a); data["r120_ok"] = (("slot",), ok)
        notes.append(f"r120 {ok.mean():.1%}")
    if "r300" in todo:
        a = np.zeros((n, len(R300_VARS), OUTER, OUTER), np.uint8); ok = np.zeros(n, bool)
        fill_source(store_index(near(args.radar300, "_300km.zarr")), slots, a, ok, radar_reader(R300_VARS))
        data["r300"] = (("slot", "c300", "y320", "x320"), a); data["r300_ok"] = (("slot",), ok)
        notes.append(f"r300 {ok.mean():.1%}")
    if "sat" in todo:
        a = np.zeros((n, len(SAT_BANDS), INNER, INNER), np.uint8); ok = np.zeros(n, bool)
        fill_source(store_index(near(args.sat, "himawari_patch_*.zarr")), slots, a, ok, sat_reader)
        data["sat"] = (("slot", "band", "y256", "x256"), a); data["sat_ok"] = (("slot",), ok)
        notes.append(f"sat {ok.mean():.1%}")
    if "ltg" in todo:
        strokes = args.strokes
        sel = strokes[(strokes["t"] > slots5[0] - STEP5) & (strokes["t"] <= slots5[-1])]
        data["ltg"] = (("slot5", "kind", "y320", "x320"), rasterize_strokes(sel, slots5))
        notes.append(f"{len(sel)} strokes")

    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    ds = xr.Dataset(data, coords={"slot": slots.astype("datetime64[ns]"), "slot5": slots5.astype("datetime64[ns]")})
    enc = {}
    for v in data:
        if v.endswith("_ok"):
            continue
        enc[v] = {"compressor": comp, "chunks": ((12,) if v == "ltg" else (1,)) + ds[v].shape[1:]}  # ltg: an hour per chunk
    attrs = {"r120_vars": list(R120_VARS), "r300_vars": list(R300_VARS), "sat_bands": list(SAT_BANDS),
             "ltg_kinds": ["IC", "CG"], "codecs": {k: list(v) for k, v in CODECS.items()},
             "center_lat": CENTER_LAT, "center_lon": CENTER_LON, "res_km": RES_KM, "offset_256_in_320": OFFSET,
             "sat_row_order": "south_to_north", "ltg_interval": "(t - 5 min, t]"}

    def write(ds: xr.Dataset, enc: dict, path: Path, mode: Literal["w", "a"]) -> None:
        try:
            ds.to_zarr(path, mode=mode, consolidated=False, zarr_format=2, encoding=enc)
        except ValueError:  # newer xarray: the zarr-python 3 spelling
            enc = {v: {"compressors": (e["compressor"],), "chunks": e["chunks"]} for v, e in enc.items()}
            ds.to_zarr(path, mode=mode, consolidated=False, zarr_format=2, encoding=enc)

    if not out.exists():
        tmp = out.with_name(out.name + ".tmp")
        ds.attrs.update(attrs)
        write(ds, enc, path=tmp, mode="w")
        tmp.rename(out)
    else:
        old = xr.open_zarr(out, consolidated=False)
        if old.sizes["slot"] != n:
            raise SystemExit(f"{out}: slot count {old.sizes['slot']} != {n}")
        for v in list(data):  # --force: replace the variable in place
            if v in old.data_vars:
                shutil.rmtree(out / v)
        write(ds.drop_vars(["slot", "slot5"]), enc, path=out, mode="a")
    return f"{year}-{month:02d}: {', '.join(notes)}, {time.time() - t0:.0f} s"


def neighbours(year: int, month: int) -> list[str]:
    prev = (year - (month == 1), 12 if month == 1 else month - 1)
    nxt = (year + (month == 12), month % 12 + 1)
    return [f"{y:04d}_{m:02d}" for y, m in (prev, (year, month), nxt)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--months", nargs=2, required=True, metavar=("FIRST", "LAST"), help="YYYY-MM YYYY-MM, inclusive")
    ap.add_argument("--radar120", type=Path, nargs="+", default=None, help="Folder(s) with YYYY_MM_120km.zarr")
    ap.add_argument("--radar300", type=Path, nargs="+", default=None, help="Folder(s) with YYYY_MM_300km.zarr")
    ap.add_argument("--sat", type=Path, nargs="+", default=None, help="Folder(s) with himawari_patch_YYYY_MM.zarr")
    ap.add_argument("--strokes-csv", type=Path, default=None, help="Needed for the ltg source")
    ap.add_argument("--sources", nargs="+", default=["r120", "r300", "sat", "ltg"], choices=["r120", "r300", "sat", "ltg"])
    ap.add_argument("--workers", type=int, default=2, help="Months packed in parallel (each needs ~6 GB of RAM)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if "ltg" in args.sources:
        if args.strokes_csv is None:
            raise SystemExit("--strokes-csv is needed for the ltg source")
        args.strokes = load_strokes(args.strokes_csv)
    first, last = (datetime.strptime(m, "%Y-%m") for m in args.months)
    months = []
    y, m = first.year, first.month
    while (y, m) <= (last.year, last.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(pack_month, *zip(*months), [args] * len(months)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
