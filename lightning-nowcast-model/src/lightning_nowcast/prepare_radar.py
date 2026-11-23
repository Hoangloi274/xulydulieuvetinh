"""Prepare the column-radar stores for experiment B from the raw SIGMET archive.

Run this where the raw volumes are (the external drive), then copy the stores to the
training server, which holds the 2025 satellite and label stores needed by build_samples.

    uv run python -m lightning_nowcast.prepare_radar \
        --raw-root ../data/2025 --out-dir data/stores/radar --start 2025-04-01T00:00 --end 2025-09-01T01:10

For each month it:
- finds the raw folder (e.g. "2025 Pro-Raw T06"), falling back to searching all of --raw-root;
- writes <out-dir>/YYYY_MM_<range>km.zarr (120 km volumes, or 300 km with --range-km 300) with lightning_nowcast.radar_zarr;
- checks the result: sorted and unique times, 10-min slot coverage and gaps, the share of the
  ROI where the -10 C level is sampled, value ranges;
- writes YYYY_MM_<range>km.done.json with those checks.

A month with a done file is skipped, so an interrupted run resumes with the same command.
The default --end includes the first hour of September: the last August anchors need it for
their auxiliary future-reflectivity targets.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .radar import GridSpec
from .radar_zarr import VARS, convert_month


def months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_raw_dir(raw_root: Path, month: int) -> Path:
    hits = [p for p in raw_root.glob(f"*T{month:02d}") if p.is_dir()]
    return hits[0] if len(hits) == 1 else raw_root


def verify(store: Path, start: datetime, end: datetime, range_km: float = 120.0) -> dict[str, Any]:
    ds = xr.open_zarr(store, consolidated=False, chunks=None)
    t = ds["time"].values.astype("datetime64[s]")
    report: dict[str, Any] = {"volumes": int(t.size), "first": str(t[0]), "last": str(t[-1]),
                                 "sorted_unique": bool(np.all(np.diff(t.astype(np.int64)) > 0)),
                                 "variables": sorted(v for v in ds.data_vars)}
    # a volume of slot T starts at about T+0 (300 km) or T+3 min (120 km): floor to 10 min gives its slot
    slots = np.unique(t.astype("datetime64[m]").astype(np.int64) // 10 * 10)
    first = np.datetime64(start, "m").astype(np.int64) // 10 * 10
    expected = np.arange(first, np.datetime64(end, "m").astype(np.int64) + 1, 10)
    missing = np.setdiff1d(expected, slots)
    gaps: list[list[str]] = []
    for e in missing:
        e_iso = str(np.datetime64(int(e), "m"))
        if gaps and np.datetime64(gaps[-1][1]) + np.timedelta64(10, "m") == np.datetime64(e_iso):
            gaps[-1][1] = e_iso
        else:
            gaps.append([e_iso, e_iso])
    report["slots_expected"] = int(expected.size)
    report["slots_missing"] = int(missing.size)
    report["gaps_over_1h"] = [g for g in gaps if np.datetime64(g[1]) - np.datetime64(g[0]) >= np.timedelta64(50, "m")]
    n = ds.sizes["x"]
    yy, xx = np.mgrid[:n, :n]
    roi = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * 2.0 <= range_km
    idx = np.unique(np.linspace(0, t.size - 1, min(60, t.size)).astype(int))
    zm10 = np.stack([ds["Z_M10"][int(i)].values for i in idx])
    cmax = np.stack([ds["CMAX"][int(i)].values for i in idx])
    vil = np.stack([ds["VIL"][int(i)].values for i in idx])
    report["Z_M10_sampled_fraction_of_roi"] = round(float(np.isfinite(zm10[:, roi]).mean()), 4)
    report["CMAX_max"] = round(float(np.nanmax(cmax)), 1)
    report["VIL_max"] = round(float(np.nanmax(vil)), 1)
    report["CMAX_ge_40_fraction"] = round(float(np.nanmean(cmax[:, roi] >= 40)), 5)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-root", type=Path, required=True, help="e.g. ../data/2025 (monthly folders '2025 Pro-Raw TMM')")
    ap.add_argument("--out-dir", type=Path, default=Path("data/stores/radar"))
    ap.add_argument("--start", type=datetime.fromisoformat, default=datetime(2025, 4, 1))
    ap.add_argument("--end", type=datetime.fromisoformat, default=datetime(2025, 9, 1, 1, 10), help="Inclusive")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--force", action="store_true", help="Rebuild months that already have a done file")
    ap.add_argument("--range-km", type=int, default=120, help="Scan task: 120 or 300 (long range)")
    ap.add_argument("--size", type=int, default=None, help="Grid cells per side (default 256 for 120 km, 320 for 300 km)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    size = args.size or (256 if args.range_km == 120 else 320)
    grid = GridSpec(nx=size, ny=size, max_range_km=float(args.range_km))

    summary: list[dict[str, Any]] = []
    for y, m in months_between(args.start, args.end):
        store = args.out_dir / f"{y:04d}_{m:02d}_{args.range_km}km.zarr"
        done = args.out_dir / f"{y:04d}_{m:02d}_{args.range_km}km.done.json"
        m_start = max(args.start, datetime(y, m, 1))
        m_end = min(args.end, (datetime(y + (m == 12), m % 12 + 1, 1) - timedelta(seconds=1)))
        if done.exists() and not args.force:
            print(f"{y}-{m:02d}: already done ({done.name}), skipping", flush=True)
            summary.append(json.loads(done.read_text()))
            continue
        raw_dir = month_raw_dir(args.raw_root, m)
        t0 = time.time()
        stats = convert_month(raw_dir, store, f"{y:04d}-{m:02d}", start=m_start, end=m_end, workers=args.workers, overwrite=True, grid=grid)
        report: dict[str, Any] = {"month": f"{y:04d}-{m:02d}", "raw_dir": str(raw_dir), "minutes": round((time.time() - t0) / 60, 1),
                  "files": stats.files, "other_tasks_skipped": stats.other_tasks_skipped,
                  "bad_volumes": len(stats.errors), "bad_volume_messages": stats.errors[:50]}
        checks = verify(store, m_start, m_end, args.range_km)
        report.update(checks)
        missing_vars = sorted(set(VARS) - set(checks["variables"]))
        if missing_vars or not checks["sorted_unique"]:
            raise SystemExit(f"{store} failed verification: missing {missing_vars}, sorted_unique={report['sorted_unique']}")
        done.write_text(json.dumps(report, indent=2))
        summary.append(report)
        print(f"{report['month']}: {report['volumes']} volumes, {report['slots_missing']}/{report['slots_expected']} slots missing, "
              f"{report['bad_volumes']} bad volumes, -10 C level sampled over {report['Z_M10_sampled_fraction_of_roi']:.1%} of the ROI "
              f"({report['minutes']} min)", flush=True)

    print("\nmonth     volumes  missing slots  bad  gaps >= 1 h")
    for r in summary:
        print(f"{r['month']}  {r['volumes']:>7}  {r['slots_missing']:>6}/{r['slots_expected']:<6} {r['bad_volumes']:>4}  {len(r['gaps_over_1h'])}")


if __name__ == "__main__":
    main()
