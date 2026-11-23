"""Retrospective shadow run: the live daemon's replay mode, day by day, with the frozen operational model.

    uv run python scripts/shadow_replay.py --work-dir shadow/2026-09-01_2026-10-23 --first 2026-09-01 --last 2026-10-23 \
        --lightning-csv data/lightning/strokes_shadow.csv [--radar-dir data/raw/radar] [--sat-raw-dir data/raw/himawari]

Each UTC day runs as one `lightning_nowcast.live --replay` over its 144 anchors, through the same code as the live run
(--no-sat-download: the replay never fetches). A day starts once its satellite is on disk (a slot of the next day exists,
or the downloader is done: --wait-pid), so it can follow a bulk download. Days already complete are skipped; the
per-day logs are <work-dir>/logs/YYYYmmdd.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ANCHORS_PER_DAY = 144


def satellite_ready(sat_dir: Path, day: date) -> bool:
    nxt = day + timedelta(days=1)
    return any((sat_dir / f"{nxt:%Y/%m/%d}").glob("*/manifest.json"))


def done(work_dir: Path, day: date) -> bool:
    return len(list((work_dir / "forecasts" / f"{day:%Y%m%d}").glob("*/forecast.json"))) >= ANCHORS_PER_DAY


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--first", type=date.fromisoformat, required=True)
    ap.add_argument("--last", type=date.fromisoformat, required=True)
    ap.add_argument("--lightning-csv", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("models/lightning_nowcast.pt"))
    ap.add_argument("--radar-dir", type=Path, default=Path("data/raw/radar"))
    ap.add_argument("--sat-raw-dir", type=Path, default=Path("data/raw/himawari"))
    ap.add_argument("--wait-pid", type=int, default=None, help="The satellite downloader: once it exits, wait no more")
    args = ap.parse_args()
    logs = args.work_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    day = args.first
    while day <= args.last:
        if done(args.work_dir, day):
            print(f"{day}: already complete", flush=True)
            day += timedelta(days=1)
            continue
        while not satellite_ready(args.sat_raw_dir, day):
            if args.wait_pid is not None and not Path(f"/proc/{args.wait_pid}").exists():
                break  # the download is over: what is missing stays missing (the model saw missing satellite in training)
            time.sleep(120)
        t0 = time.time()
        first, last = datetime.combine(day, datetime.min.time()), datetime.combine(day, datetime.min.time()) + timedelta(minutes=10 * (ANCHORS_PER_DAY - 1))
        cmd = [sys.executable, "-m", "lightning_nowcast.live", "--ckpt", str(args.ckpt), "--work-dir", str(args.work_dir),
               "--radar-dir", str(args.radar_dir), "--sat-raw-dir", str(args.sat_raw_dir), "--no-sat-download",
               "--lightning-csv", str(args.lightning_csv), "--replay", first.isoformat(), last.isoformat()]
        with (logs / f"{day:%Y%m%d}.log").open("w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
        n = len(list((args.work_dir / "forecasts" / f"{day:%Y%m%d}").glob("*/forecast.json")))
        print(f"{day}: {'ok' if r.returncode == 0 else f'FAILED ({r.returncode})'}, {n} forecasts, {time.time() - t0:.0f} s", flush=True)
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
