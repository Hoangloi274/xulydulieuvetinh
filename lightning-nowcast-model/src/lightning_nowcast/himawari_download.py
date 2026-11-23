import argparse
import bz2
import io
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config

# Print immediately even when stdout is a pipe (the live daemons read it line by line).
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)

# ----------------------------
# Config
# ----------------------------
BANDS = ["B08", "B10", "B13", "B15", "B16"]
SEGMENTS = ["S0410", "S0510"]
PRODUCT = "AHI-L1b-FLDK"
RES_CODE = "R20"
FALLBACK_SOURCES = [("noaa-himawari9", "H09"), ("noaa-himawari8", "H08")]

EXPECTED_HOUSEKEEPING_TIMES = {(2, 40), (14, 40)}

# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class SourceSpec:
    bucket: str
    satellite: str


@dataclass(frozen=True)
class DownloadSpec:
    bucket: str
    key: str
    out_path: Path


# ----------------------------
# S3 helpers
# ----------------------------
def make_s3_client(max_workers: int = 10):
    # The default pool (10) is smaller than --max-workers, which forces connection churn.
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, max_pool_connections=max(10, max_workers)),
    )


def time_prefix(dt: datetime) -> str:
    return f"{PRODUCT}/{dt:%Y/%m/%d/%H%M}/"


def list_slot_objects(s3, bucket: str, dt: datetime) -> list[str]:
    # One slot holds ~160 keys (16 bands x 10 segments): a single request, independent
    # of how far into the day we are. Listing the whole day prefix instead costs ~23
    # sequential requests per bucket by late UTC evening, on every invocation.
    prefix = time_prefix(dt)
    paginator = s3.get_paginator("list_objects_v2")
    out: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                out.append(key)
    return out


# ----------------------------
# Timestamp planning
# ----------------------------
def iter_times_10min(start: datetime, end: datetime) -> Iterable[datetime]:
    dt = start
    while dt < end:
        yield dt
        dt += timedelta(minutes=10)


def expected_housekeeping_miss(dt: datetime) -> bool:
    return (dt.hour, dt.minute) in EXPECTED_HOUSEKEEPING_TIMES


def expected_name(satellite: str, dt: datetime, band: str, segment: str) -> str:
    return f"HS_{satellite}_{dt:%Y%m%d_%H%M}_{band}_FLDK_{RES_CODE}_{segment}.DAT.bz2"


def build_expected_names(satellite: str, dt: datetime) -> list[str]:
    names: list[str] = []
    for band in BANDS:
        for seg in SEGMENTS:
            names.append(expected_name(satellite, dt, band, seg))
    return names


def choose_source_for_timestamp(s3, dt: datetime) -> tuple[SourceSpec | None, list[str]]:
    # Fallback buckets are only listed when the preferred one is incomplete.
    for bucket, satellite in FALLBACK_SOURCES:
        key_map = {Path(k).name: k for k in list_slot_objects(s3, bucket, dt)}
        names = build_expected_names(satellite, dt)
        if all(name in key_map for name in names):
            return SourceSpec(bucket=bucket, satellite=satellite), [key_map[n] for n in names]
    return None, []


# ----------------------------
# Validation
# ----------------------------
def validate_bz2_file(path: Path) -> None:
    with bz2.open(path, "rb") as f:
        while f.read(1024 * 1024):
            pass


def validate_with_satpy(files: Sequence[Path]) -> None:
    from satpy import Scene

    scn = Scene(reader="ahi_hsd", filenames=[str(p) for p in files])
    scn.load(BANDS, calibration="brightness_temperature", pad_data=False)
    if scn.missing_datasets:
        raise RuntimeError(f"Satpy missing datasets: {scn.missing_datasets}")


# ----------------------------
# Download helpers
# ----------------------------
def atomic_download(s3, spec: DownloadSpec, retries: int = 2) -> None:
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = spec.out_path.with_suffix(spec.out_path.suffix + ".part")

    last_err: Exception | None = None
    for _ in range(retries + 1):
        if part_path.exists():
            part_path.unlink()
        try:
            s3.download_file(spec.bucket, spec.key, str(part_path))
            validate_bz2_file(part_path)
            os.replace(part_path, spec.out_path)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            if part_path.exists():
                part_path.unlink()
    assert last_err is not None
    raise last_err


def timestamp_dir(out_root: Path, dt: datetime) -> Path:
    return out_root / f"{dt:%Y}" / f"{dt:%m}" / f"{dt:%d}" / f"{dt:%H%M}"


def is_complete(ts_dir: Path) -> bool:
    """A slot is complete iff its manifest exists (written only after Satpy validation)
    and every listed file is still on disk."""
    manifest_path = ts_dir / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return False
    files = manifest.get("files") or []
    if len(files) != len(BANDS) * len(SEGMENTS):
        return False
    return all((ts_dir / name).is_file() and (ts_dir / name).stat().st_size > 0 for name in files)


def write_manifest(ts_dir: Path, dt: datetime, source: SourceSpec, strict: bool, files: Sequence[Path]) -> None:
    manifest = {
        "time_utc": dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "bucket": source.bucket,
        "satellite": source.satellite,
        "bands": BANDS,
        "segments": SEGMENTS,
        "strict": strict,
        "files": [p.name for p in files],
    }
    with (ts_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ----------------------------
# Per-timestamp processing
# ----------------------------
def process_timestamp(
    s3,
    out_root: Path,
    dt: datetime,
    source: SourceSpec,
    keys: Sequence[str],
    strict: bool,
    max_workers: int,
    satpy_retries: int,
) -> tuple[str, str]:
    ts_dir = timestamp_dir(out_root, dt)
    if ts_dir.exists():
        shutil.rmtree(ts_dir)
    ts_dir.mkdir(parents=True, exist_ok=True)

    specs = [DownloadSpec(bucket=source.bucket, key=key, out_path=ts_dir / Path(key).name) for key in keys]
    files = [spec.out_path for spec in specs]

    def download_all() -> None:
        for p in files:
            if p.exists():
                p.unlink()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(atomic_download, s3, spec): spec for spec in specs}
            for fut in as_completed(futs):
                fut.result()

    try:
        last_err: Exception | None = None
        for attempt in range(satpy_retries + 1):
            # A semantic validation failure may still be due to a bad-but-bz2-valid file,
            # so each retry re-downloads the full timestamp.
            download_all()
            try:
                validate_with_satpy(files)
                write_manifest(ts_dir, dt, source, strict, files)
                return "ok", f"ok   {dt:%Y-%m-%d %H:%M} ({source.satellite})"
            except Exception as e:  # noqa: BLE001
                last_err = e

        assert last_err is not None
        raise last_err

    except Exception as e:  # noqa: BLE001
        if ts_dir.exists():
            shutil.rmtree(ts_dir)
        return "fail", f"fail {dt:%Y-%m-%d %H:%M} ({type(e).__name__}: {e})"


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download Himawari HSD locally with BZip2 + Satpy validation.")
    ap.add_argument("--start", type=str, required=True, help="UTC start, e.g. 2025-04-01T00:00")
    ap.add_argument("--end", type=str, required=True, help="UTC end (exclusive)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=16, help="Per-timestamp parallel file downloads")
    ap.add_argument("--slot-workers", type=int, default=1, help="Timestamps downloaded at the same time (bulk/archive downloads)")
    ap.add_argument("--strict", action="store_true", help="Reserved for manifest metadata only")
    ap.add_argument("--satpy-retries", type=int, default=1, help="How many times to redownload a timestamp after Satpy validation fails")
    ap.add_argument("--force", action="store_true", help="Re-download timestamps that are already complete on disk")
    ap.add_argument(
        "--wait-latest",
        type=int,
        default=0,
        help="Seconds to keep polling for the last timestamp in the range if it is not yet on S3 "
        "(cheap single-slot LIST, no process restart). 0 = no waiting.",
    )
    ap.add_argument("--poll-interval", type=int, default=20, help="Seconds between polls for --wait-latest")
    return ap.parse_args()


def parse_iso_minute(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def main() -> int:
    args = parse_args()
    start = parse_iso_minute(args.start)
    end = parse_iso_minute(args.end)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    s3 = make_s3_client(args.max_workers * args.slot_workers)

    times = list(iter_times_10min(start, end))

    def handle(i: int, dt: datetime) -> tuple[str, str]:
        """Download one slot; returns (kind, message) with kind in ok/cached/miss/fail."""
        if not args.force and is_complete(timestamp_dir(out_root, dt)):
            # Already downloaded and Satpy-validated by an earlier run; report it as ok
            # so callers that grep for "ok <slot>" keep working.
            return "cached", f"ok   {dt:%Y-%m-%d %H:%M} (cached)"
        source, keys = choose_source_for_timestamp(s3, dt)
        is_last = i == len(times) - 1
        if source is None and is_last and args.wait_latest > 0 and not expected_housekeeping_miss(dt):
            deadline = time.monotonic() + args.wait_latest
            while source is None and time.monotonic() < deadline:
                print(f"wait {dt:%Y-%m-%d %H:%M} (not yet complete on S3)")
                time.sleep(max(1, min(args.poll_interval, deadline - time.monotonic())))
                source, keys = choose_source_for_timestamp(s3, dt)
        if source is None:
            if expected_housekeeping_miss(dt):
                return "miss", f"expected_housekeeping_miss {dt:%Y-%m-%d %H:%M}"
            return "miss", f"miss {dt:%Y-%m-%d %H:%M} (no suitable files in H09/H08)"
        status, msg = process_timestamp(
            s3=s3,
            out_root=out_root,
            dt=dt,
            source=source,
            keys=keys,
            strict=args.strict,
            max_workers=args.max_workers,
            satpy_retries=args.satpy_retries,
        )
        return (status if status in ("ok", "fail") else "miss"), msg

    counts = {"ok": 0, "cached": 0, "miss": 0, "fail": 0}
    # Slots are independent: several are fetched at once (--slot-workers); messages print in time order.
    with ThreadPoolExecutor(max_workers=max(1, args.slot_workers)) as pool:
        for kind, msg in pool.map(lambda a: handle(*a), enumerate(times)):
            print(msg)
            counts[kind] += 1
    ok = counts["ok"] + counts["cached"]
    skip, miss, fail = counts["cached"], counts["miss"], counts["fail"]

    print(f"summary: ok={ok} (cached={skip}) miss={miss} fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())