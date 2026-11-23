import argparse
import contextlib
import json
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from numcodecs import Blosc
from pyresample import create_area_def

warnings.filterwarnings("ignore", category=RuntimeWarning)
with contextlib.suppress(ImportError):  # older zarr has no ZarrUserWarning
    from zarr.errors import ZarrUserWarning

    warnings.filterwarnings("ignore", category=ZarrUserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

BANDS_DEFAULT = ["B08", "B10", "B13", "B15", "B16"]
FILL = np.uint16(65535)
SCALE = 100.0  # store Kelvin * 100


@dataclass(frozen=True)
class GridSpec:
    center_lat: float
    center_lon: float
    nx: int
    ny: int
    res_m: int


@dataclass(frozen=True)
class WorkerConfig:
    bands: tuple[str, ...]
    grid: GridSpec
    cache_root: str | None
    no_cache: bool


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1)


def next_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return datetime(dt.year + 1, 1, 1)
    return datetime(dt.year, dt.month + 1, 1)


def iter_days(t0: datetime, t1: datetime):
    d = date(t0.year, t0.month, t0.day)
    d1 = date(t1.year, t1.month, t1.day)
    while d <= d1:
        yield d
        d = d + timedelta(days=1)


def area_from_grid(grid: GridSpec):
    half_x = (grid.nx * grid.res_m) // 2
    half_y = (grid.ny * grid.res_m) // 2
    return create_area_def(
        area_id="nhabe_aeqd",
        projection={
            "proj": "aeqd",
            "lat_0": grid.center_lat,
            "lon_0": grid.center_lon,
            "units": "m",
        },
        width=grid.nx,
        height=grid.ny,
        area_extent=(-half_x, -half_y, half_x, half_y),
    )


def timestamp_dirs_for_day(
    raw_root: Path, day: date, t0: datetime, t1: datetime
) -> list[Path]:
    """Complete timestamp folders of `day` whose slot time lies in [t0, t1)."""
    day_dir = raw_root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    if not day_dir.exists():
        return []
    out: list[Path] = []
    for child in day_dir.iterdir():
        if not (
            child.is_dir()
            and len(child.name) == 4
            and child.name.isdigit()
            and (child / "manifest.json").exists()
        ):
            continue
        slot = datetime(day.year, day.month, day.day, int(child.name[:2]), int(child.name[2:]))
        if t0 <= slot < t1:
            out.append(child)
    return sorted(out)


def validate_and_get_files(
    ts_dir: Path, bands: tuple[str, ...]
) -> tuple[datetime, int, list[Path]]:
    manifest_path = ts_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    dt = datetime.fromisoformat(manifest["time_utc"])
    sat = str(manifest.get("satellite", ""))
    sat_code = 9 if sat == "H09" else 8 if sat == "H08" else 0

    manifest_files = [ts_dir / name for name in manifest.get("files", [])]
    if not manifest_files:
        raise FileNotFoundError("manifest contains no files")

    selected: list[Path] = []
    for p in manifest_files:
        if not p.exists():
            raise FileNotFoundError(f"missing manifest file: {p.name}")
        if any(f"_{band}_" in p.name for band in bands):
            selected.append(p)

    needed = {band: 0 for band in bands}
    for p in selected:
        for band in bands:
            if f"_{band}_" in p.name:
                needed[band] += 1
    missing = [band for band, n in needed.items() if n == 0]
    if missing:
        raise FileNotFoundError(f"manifest has no files for bands: {missing}")

    return dt, sat_code, selected


def to_uint16_bt(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    scaled = np.rint(data * SCALE)
    # clip first: clipping after the fill turned every NaN into 65534, which is not masked on read
    scaled = np.where(np.isfinite(scaled), np.clip(scaled, 0, 65534), FILL).astype(np.uint16)
    return scaled


def process_timestamp(ts_dir_str: str, cfg: WorkerConfig) -> dict[str, Any]:
    from satpy import Scene

    ts_dir = Path(ts_dir_str)
    dt, sat_code, files = validate_and_get_files(ts_dir, cfg.bands)
    area = area_from_grid(cfg.grid)

    filenames = [str(p) for p in files]
    scn = Scene(reader="ahi_hsd", filenames=filenames)
    scn.load(list(cfg.bands), calibration="brightness_temperature", pad_data=False)
    if scn.missing_datasets:
        raise RuntimeError(f"missing datasets: {scn.missing_datasets}")

    if cfg.no_cache:
        rs = scn.resample(area, resampler="nearest")
    else:
        assert cfg.cache_root is not None
        worker_cache = Path(cfg.cache_root) / f"worker_{os.getpid()}"
        worker_cache.mkdir(parents=True, exist_ok=True)
        rs = scn.resample(area, resampler="nearest", cache_dir=str(worker_cache))

    arrays: dict[str, np.ndarray] = {}
    for band in cfg.bands:
        patch = rs[band].compute().data
        arrays[band] = to_uint16_bt(patch)

    return {
        "dt": np.datetime64(dt),
        "satellite": np.uint8(sat_code),
        "arrays": arrays,
    }


def open_last_time_if_exists(zarr_path: Path) -> np.datetime64 | None:
    if not zarr_path.exists():
        return None
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        if "time" not in ds or ds.sizes.get("time", 0) == 0:
            return None
        return ds["time"].values[-1]
    except Exception:  # noqa: BLE001  (a store being written or damaged: start over)
        return None


def write_batch(
    zarr_path: Path, batch: list[dict[str, Any]], cfg: WorkerConfig, first_write: bool
) -> None:
    if not batch:
        return

    batch = sorted(batch, key=lambda r: r["dt"])
    grid = cfg.grid
    x = (np.arange(grid.nx) - grid.nx / 2 + 0.5) * grid.res_m
    y = (np.arange(grid.ny) - grid.ny / 2 + 0.5) * grid.res_m

    data_vars: dict[str, Any] = {}
    for band in cfg.bands:
        data_vars[band] = (
            ("time", "y", "x"),
            np.stack([rec["arrays"][band] for rec in batch], axis=0),
        )
    data_vars["satellite"] = (
        ("time",),
        np.array([rec["satellite"] for rec in batch], dtype=np.uint8),
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.array([rec["dt"] for rec in batch]).astype("datetime64[ns]"),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
        },
        attrs={
            "center_lat": grid.center_lat,
            "center_lon": grid.center_lon,
            "projection": "aeqd",
            "res_m": grid.res_m,
            "scale_K_x100": SCALE,
            "fill_value_uint16": int(FILL),
            "bands": ",".join(cfg.bands),
            "note_satellite": "satellite=9 => Himawari-9; satellite=8 => Himawari-8",
            # Rows are pyresample image rows, north -> south; the ascending `y` coordinate does not
            # describe them (radar and label stores run south -> north). Kept for compatibility
            # with existing stores and models; lightning_nowcast.pack reads this attribute and orients the rows.
            "row_order": "north_to_south",
        },
    )

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    if first_write:
        encoding = {
            band: {
                "dtype": "uint16",
                "_FillValue": FILL,
                "chunks": (1, grid.ny, grid.nx),
                "compressor": compressor,
            }
            for band in cfg.bands
        }
        encoding["satellite"] = {
            "dtype": "uint8",
            "chunks": (1024,),
            "compressor": compressor,
        }
        ds.to_zarr(
            str(zarr_path),
            mode="w",
            append_dim=None,
            consolidated=False,
            zarr_format=2,
            encoding=encoding,
        )
    else:
        ds.to_zarr(
            str(zarr_path),
            mode="a",
            append_dim="time",
            consolidated=False,
            zarr_format=2,
        )


def process_day(
    day: date, t0: datetime, t1: datetime, raw_root: Path, cfg: WorkerConfig, workers: int
) -> tuple[list[dict[str, Any]], int, int]:
    ts_dirs = timestamp_dirs_for_day(raw_root, day, t0, t1)
    if not ts_dirs:
        return [], 0, 0

    ok: list[dict[str, Any]] = []
    fail = 0

    if workers <= 1:
        for ts_dir in ts_dirs:
            try:
                ok.append(process_timestamp(str(ts_dir), cfg))
            except Exception as e:  # noqa: BLE001  (one bad slot must not stop the month; logged)
                fail += 1
                print(f"fail {ts_dir.relative_to(raw_root)} ({type(e).__name__}: {e})")
        return ok, len(ok), fail

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(process_timestamp, str(ts_dir), cfg): ts_dir for ts_dir in ts_dirs
        }
        for fut in as_completed(futs):
            ts_dir = futs[fut]
            try:
                ok.append(fut.result())
            except Exception as e:  # noqa: BLE001  (one bad slot must not stop the month; logged)
                fail += 1
                print(f"fail {ts_dir.relative_to(raw_root)} ({type(e).__name__}: {e})")
    return ok, len(ok), fail


def maybe_remove(zarr_path: Path, overwrite: bool) -> None:
    if overwrite and zarr_path.exists():
        shutil.rmtree(zarr_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert raw local Himawari timestamp folders directly to monthly Zarr."
    )
    ap.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Root of local Himawari raw folders: YYYY/MM/DD/HHMM",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write monthly Zarr stores",
    )
    ap.add_argument(
        "--start",
        type=str,
        required=True,
        help="Inclusive start time, e.g. 2025-04-01T00:00",
    )
    ap.add_argument(
        "--end",
        type=str,
        required=True,
        help="Exclusive end time, e.g. 2025-12-01T00:00",
    )
    ap.add_argument("--center-lat", type=float, required=True)
    ap.add_argument("--center-lon", type=float, required=True)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--res-m", type=int, default=2000)
    ap.add_argument("--bands", nargs="+", default=BANDS_DEFAULT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-root", type=Path, default=Path("./satpy_resample_cache"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument(
        "--overwrite-month",
        action="store_true",
        help="Overwrite each monthly Zarr if it already exists",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing monthly Zarr from its last stored time",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_cache:
        args.cache_root.mkdir(parents=True, exist_ok=True)

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    if not start < end:
        raise ValueError("--start must be earlier than --end")

    cfg = WorkerConfig(
        bands=tuple(args.bands),
        grid=GridSpec(
            center_lat=args.center_lat,
            center_lon=args.center_lon,
            nx=args.nx,
            ny=args.ny,
            res_m=args.res_m,
        ),
        cache_root=None if args.no_cache else str(args.cache_root),
        no_cache=bool(args.no_cache),
    )

    m0 = month_start(start)
    while m0 < end:
        m1 = next_month(m0)
        month_begin = max(start, m0)
        month_end = min(end, m1)
        zarr_path = args.out_dir / f"himawari_patch_{m0:%Y_%m}.zarr"

        if args.overwrite_month:
            maybe_remove(zarr_path, overwrite=True)

        resume_after: datetime | None = None
        first_write = not zarr_path.exists()
        if args.resume and zarr_path.exists():
            last = open_last_time_if_exists(zarr_path)
            if last is not None:
                last_dt = last.astype("datetime64[s]").astype(datetime)
                resume_after = last_dt + timedelta(minutes=10)
                first_write = False

        month_ok = 0
        month_fail = 0
        wrote_any = False

        if resume_after is not None:
            month_begin = max(month_begin, resume_after)

        for day in iter_days(month_begin, month_end - timedelta(seconds=1)):
            # Only slots in [month_begin, month_end) are converted, so the raw
            # archive can hold far more than the requested window.
            batch, ok_count, fail_count = process_day(
                day, month_begin, month_end, args.raw_dir, cfg, workers=args.workers
            )
            if batch:
                write_batch(zarr_path, batch, cfg, first_write=first_write)
                first_write = False
                wrote_any = True
            month_ok += ok_count
            month_fail += fail_count
            print(f"[{day}] ok={ok_count} fail={fail_count}")

        print(
            f"[{m0:%Y-%m}] wrote={wrote_any} ok={month_ok} fail={month_fail} -> {zarr_path}"
        )
        m0 = m1


if __name__ == "__main__":
    main()
