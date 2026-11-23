"""Convert raw Nha Be SIGMET/IRIS volumes of one month to a Zarr store with column radar fields.

    uv run python -m lightning_nowcast.radar_zarr \
        --raw-root /path/to/sigmet --out data/stores/radar/2025_06_120km.zarr --month 2025-06 --workers 16

The store holds Z and Q exactly as scripts/radar_sigmet_to_zarr.py writes them (lowest sweep),
plus CMAX, Z_M10, Z_M20, ET30 and VIL (see radar.py), with the same time coordinate (the
volume start time from the file name). Volumes of the other task are skipped.

Nha Be alternates two tasks every 10 min: a 300 km long-range volume (4 sweeps, 0.5-3.6 deg,
starting at about T+0) and the 120 km volume (8 sweeps, starting at about T+3). --range-km picks
the task; --size sets the grid (2 km cells), e.g. 256 for 120 km and 320 for 300 km. The two
grids share their cell centres, so a 256 grid sits at offset 32 inside a 320 grid.
"""
from __future__ import annotations

import argparse
import re
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Blosc

from .radar import FIELDS_3D, GridSpec, volume_features

FILENAME_RE = re.compile(r"^NHB(?P<ts>\d{12})\.RAW")


def parse_time_from_name(path: Path) -> datetime | None:
    """Volume start time (UTC) from names like NHB250601000303.RAWTVG5."""
    m = FILENAME_RE.match(path.name)
    return datetime.strptime("20" + m.group("ts"), "%Y%m%d%H%M%S") if m else None


def task_of_name(path: Path) -> int | None:
    """300 or 120 from the start minute in the file name (300 km volumes start at :x0, 120 km at :x3), else None."""
    t = parse_time_from_name(path)
    if t is None:
        return None
    return 300 if t.minute % 10 <= 1 else 120 if 2 <= t.minute % 10 <= 5 else None


def iter_month_files(raw_root: Path, month: str) -> list[Path]:
    """Raw volumes of YYYY-MM under raw_root (recursive), one per file name, in time order."""
    year, mon = month.split("-")
    by_name: dict[str, Path] = {}
    for p in sorted(raw_root.rglob(f"NHB{year[2:]}{mon}*.RAW*")):
        if p.is_file():
            by_name.setdefault(p.name, p)
    return sorted(by_name.values(), key=lambda p: p.name)

VARS = ("Z", "Q") + FIELDS_3D


def work(path: Path, grid: GridSpec, h_m10_m: float, h_m20_m: float) -> tuple[datetime, dict[str, np.ndarray] | None, str]:
    warnings.filterwarnings("ignore")
    t = parse_time_from_name(path)
    assert t is not None
    try:
        return t, volume_features(path, grid, h_m10_m=h_m10_m, h_m20_m=h_m20_m), ""
    except Exception as exc:  # noqa: BLE001  (a corrupt volume must not stop the month)
        return t, None, f"{path.name}: {type(exc).__name__}: {exc}"


@dataclass
class MonthStats:
    files: int = 0
    volumes: int = 0
    other_tasks_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def convert_month(raw_root: Path, out: Path, month: str, *, start: datetime | None = None, end: datetime | None = None,
                  h_m10_m: float = 6500.0, h_m20_m: float = 8000.0, workers: int = 8, batch: int = 256,
                  overwrite: bool = False, grid: GridSpec | None = None) -> MonthStats:
    """Write one month (optionally limited to [start, end]) to `out`; returns counts and error messages."""
    files = [p for p in iter_month_files(raw_root, month)
             if (t := parse_time_from_name(p)) is not None
             and (start is None or t >= start) and (end is None or t <= end)]
    grid = grid or GridSpec()
    # the file name tells the task, so volumes of the other task are not even opened (read_volume still checks the range)
    n_other = sum(1 for p in files if task_of_name(p) not in (None, round(grid.max_range_km)))
    files = [p for p in files if task_of_name(p) in (None, round(grid.max_range_km))]
    if not files:
        raise FileNotFoundError(f"no volumes for {month} under {raw_root}")
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to rebuild it")
        shutil.rmtree(out)
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    first, n_ok, n_skip = True, 0, n_other
    errors: list[str] = []
    print(f"{month}: {len(files)} files under {raw_root}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        # one ordered stream over the month: workers keep reading while the main process appends to the store
        stream = ex.map(work, files, [grid] * len(files), [h_m10_m] * len(files), [h_m20_m] * len(files), chunksize=4)
        for i0 in range(0, len(files), batch):
            res = [next(stream) for _ in files[i0:i0 + batch]]
            errors += [e for _, _, e in res if e]
            good = sorted((t, f) for t, f, _ in res if f is not None)
            n_skip += sum(1 for _, f, e in res if f is None and not e)
            if not good:
                continue
            ds = xr.Dataset(
                {v: (("time", "y", "x"), np.stack([f[v] for _, f in good]).astype(np.float32)) for v in VARS},
                coords={"time": np.array([t for t, _ in good], dtype="datetime64[ns]"), "y": grid.y, "x": grid.x},
                attrs={"projection": "local_aeqd_like", "center_lat": grid.center_lat, "center_lon": grid.center_lon,
                       "res_m": grid.res_m, "max_range_km": grid.max_range_km,
                       "height_m10_m": h_m10_m, "height_m20_m": h_m20_m,
                       "notes": "Z/Q: lowest sweep as in radar_sigmet_to_zarr.py; CMAX, Z_M10, Z_M20 dBZ; ET30 km; VIL kg m-2; "
                                "NaN = not sampled (Z_M10/Z_M20 are unsampled within ~30 km of the radar)."},
            )
            if first:
                enc = {v: {"dtype": "float32", "chunks": (1, grid.ny, grid.nx), "compressor": comp} for v in VARS}
                ds.to_zarr(out, mode="w", consolidated=False, zarr_format=2, encoding=enc)
                first = False
            else:
                ds.to_zarr(out, mode="a", append_dim="time", consolidated=False, zarr_format=2)
            n_ok += len(good)
            print(f"  [{min(i0 + batch, len(files))}/{len(files)}] volumes written {n_ok}, other tasks skipped {n_skip}, errors {len(errors)}", flush=True)
    return MonthStats(len(files) + n_other, n_ok, n_skip, errors)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-root", type=Path, required=True, help="Directory searched recursively for NHB*.RAW* files")
    ap.add_argument("--out", type=Path, required=True, help="Output store, e.g. data/stores/radar/2025_06_120km.zarr")
    ap.add_argument("--month", type=str, required=True, help="YYYY-MM")
    ap.add_argument("--start", type=datetime.fromisoformat, default=None, help="Optional inclusive start (UTC)")
    ap.add_argument("--end", type=datetime.fromisoformat, default=None, help="Optional inclusive end (UTC)")
    ap.add_argument("--height-m10-m", type=float, default=6500.0, help="Height of the -10 C level (tropical wet season ~6.5 km)")
    ap.add_argument("--height-m20-m", type=float, default=8000.0, help="Height of the -20 C level")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256, help="Volumes per append")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--range-km", type=float, default=120.0, help="Scan task: 120 (8 sweeps) or 300 (long range, 4 sweeps)")
    ap.add_argument("--size", type=int, default=256, help="Grid cells per side (2 km): 256 for 120 km, 320 for 300 km")
    args = ap.parse_args()
    try:
        stats = convert_month(args.raw_root, args.out, args.month, start=args.start, end=args.end, h_m10_m=args.height_m10_m,
                              h_m20_m=args.height_m20_m, workers=args.workers, batch=args.batch, overwrite=args.overwrite,
                              grid=GridSpec(nx=args.size, ny=args.size, max_range_km=args.range_km))
    except (FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
    for e in stats.errors[:20]:
        print("error", e)
    print(f"done: {args.out} with {stats.volumes} volumes")


if __name__ == "__main__":
    main()
