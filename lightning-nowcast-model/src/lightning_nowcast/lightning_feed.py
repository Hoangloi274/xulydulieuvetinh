"""iWeather lightning feed: archive the 5-min stroke frames, live or for a past period, and read them back as strokes.

Adapted from the Đài's download tool (set_iweather_app.py), which produced the training CSV
(data/lightning/strokes_2025-01_2026-09_300km.csv): same endpoint, login, row format and 300 km filter,
so live strokes match the training data. Credentials come from the environment (IWEATHER_USER, IWEATHER_PASS).

    # once: how fresh is the feed? (newest stroke age, strokes per frame)
    uv run python -m lightning_nowcast.lightning_feed probe --root data/lightning/feed
    # live: poll every minute, refetch frames until they settle, log the latency
    uv run python -m lightning_nowcast.lightning_feed live --root data/lightning/feed
    # a past period (UTC), optionally exported in the training CSV layout
    uv run python -m lightning_nowcast.lightning_feed backfill --root data/lightning/feed \
        --start 2026-10-01T00:00 --end 2026-10-20T00:00 --csv data/lightning/strokes_2026-10.csv

Archive: <root>/raw/YYYY/MM/DD/<frame UTC YYYYmmddHHMM>.tsv, the rows as returned (tab-separated). A frame younger
than --settle-min is written as .partial and fetched again on later polls, because strokes keep arriving after its
nominal time; after that it is final and never fetched again. <root>/latency.jsonl logs every live poll,
including the delivery delay of the strokes that appeared since the last poll (first seen - stroke time), which
is the latency the model's lightning input must be trained with. Strokes already there at start-up are not timed.
Frames are requested by their nominal time in Vietnam time (UTC+7), as the website does.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

BASE = "https://iweather.gov.vn/product/lightning"
LOGIN_URL = "https://iweather.gov.vn/auth/login"
VN = timedelta(hours=7)
STEP = timedelta(minutes=5)
CENTER_LAT, CENTER_LON, RADIUS_KM = 10.68, 106.75, 300.0  # the download tool's defaults (training CSV)
CSV_HEAD = ["time_utc", "time_vn", "lat", "lon", "dist_km"] + [f"c{i}" for i in range(11, 26)]
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


class TokenError(Exception):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def _headers() -> dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*", "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
            "Referer": "https://iweather.gov.vn/dashboard", "Origin": "https://iweather.gov.vn"}


def login(user: str, password: str) -> str:
    """POST the login form (JSON first, form-encoded as fallback) and return the JWT found in the reply."""
    payload = {"identifier": user, "password": password}
    for kind in ("json", "form"):
        h = _headers()
        if kind == "json":
            body, h["Content-Type"] = json.dumps(payload).encode(), "application/json"
        else:
            body, h["Content-Type"] = urllib.parse.urlencode(payload).encode(), "application/x-www-form-urlencoded"
        try:
            with urllib.request.urlopen(urllib.request.Request(LOGIN_URL, data=body, headers=h, method="POST"), timeout=30) as r:
                reply = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if kind == "json" and e.code in (400, 415):
                continue
            raise RuntimeError(f"login failed: HTTP {e.code}") from e
        m = _JWT.search(reply)
        if not m:
            raise RuntimeError("login reply holds no token")
        return m.group(0)
    raise RuntimeError("login failed: neither JSON nor form accepted")


class Client:
    """Fetches frames, logging in again when the token expires."""

    def __init__(self, user: str | None = None, password: str | None = None) -> None:
        self.user = user or os.environ.get("IWEATHER_USER", "")
        self.password = password or os.environ.get("IWEATHER_PASS", "")
        if not (self.user and self.password):
            raise SystemExit("set IWEATHER_USER and IWEATHER_PASS (the Đài's iWeather account)")
        self.token: str | None = None

    def frame(self, t_utc: datetime, retries: int = 3) -> list[list[str]]:
        """Rows of the frame with nominal time t_utc (requested as Vietnam time)."""
        for attempt in range(2):
            if self.token is None:
                self.token = login(self.user, self.password)
            try:
                return parse(self._get(t_utc + VN, retries))
            except TokenError:
                self.token = None
                if attempt:
                    raise
        return []

    def _get(self, t_vn: datetime, retries: int) -> str:
        url = f"{BASE}?token={self.token}&time={t_vn:%Y%m%d%H%M}"
        err = ""
        for k in range(retries):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=30) as r:
                    return r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise TokenError(f"HTTP {e.code}") from e
                err = f"HTTP {e.code}"
            except Exception as e:  # noqa: BLE001  (network hiccup: retry)
                err = str(e)
            time.sleep(2 * (k + 1))
        raise RuntimeError(f"frame {t_vn:%Y-%m-%d %H:%M} VN: {err}")


def parse(text: str) -> list[list[str]]:
    """Rows of >= 10 whitespace/tab separated fields; a JSON reply mentioning a token means it expired."""
    s = text.strip()
    if s.startswith(("{", "[")):
        if "token" in s.lower() or "message" in s.lower():
            raise TokenError(s[:200])
        return []
    return [p for line in s.splitlines() if len(p := (line.split("\t") if "\t" in line else line.split())) >= 10]


def stroke_time_utc(row: list[str]) -> datetime:
    y, mo, d, h, mi, se = (int(row[i]) for i in range(1, 7))
    return datetime(y, mo, d, h, mi, se, int(row[7]) // 1000)


def floor5(t: datetime) -> datetime:
    return t.replace(minute=t.minute - t.minute % 5, second=0, microsecond=0)


def frame_times(start: datetime, end: datetime) -> Iterator[datetime]:
    t = floor5(start)
    while t < end:
        yield t
        t += STEP


def frame_path(root: Path, t: datetime, partial: bool = False) -> Path:
    return root / "raw" / f"{t:%Y/%m/%d}" / f"{t:%Y%m%d%H%M}.tsv{'.partial' if partial else ''}"


def write_frame(root: Path, t: datetime, rows: list[list[str]], final: bool) -> None:
    p = frame_path(root, t, partial=not final)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join("\t".join(r) for r in rows), encoding="utf-8")
    tmp.replace(p)
    if final:
        frame_path(root, t, partial=True).unlink(missing_ok=True)


def sync(client: Client, root: Path, start: datetime, end: datetime, settle_min: float, delay: float = 0.3,
         log: Callable[[str], None] = print, workers: int = 1) -> dict[datetime, list[list[str]]]:
    """Fetch every frame in [start, end) that is not final yet; returns the rows fetched now.
    workers > 1 requests frames concurrently (one round trip for a few frames; `delay` is then not used)."""
    now = utcnow()
    todo = [t for t in frame_times(start, end) if not frame_path(root, t).exists()]

    def fetch(t: datetime) -> list[list[str]]:
        rows = client.frame(t)
        write_frame(root, t, rows, final=now - (t + STEP) >= timedelta(minutes=settle_min))
        return rows

    got: dict[datetime, list[list[str]]] = {}
    if workers > 1 and len(todo) > 1:
        if client.token is None:
            client.token = login(client.user, client.password)  # once, not per thread
        with ThreadPoolExecutor(min(workers, len(todo))) as ex:
            got = dict(zip(todo, ex.map(fetch, todo)))
    else:
        for t in todo:
            got[t] = fetch(t)
            if delay:
                time.sleep(delay)
    if got:
        log(f"fetched {len(got)} frames {min(got):%Y-%m-%d %H:%M} .. {max(got):%H:%M} UTC")
    return got


def read_rows(root: Path, start: datetime, end: datetime) -> list[list[str]]:
    """Unique rows of the archived frames that may hold strokes in [start, end) (final or partial)."""
    seen: set[tuple] = set()
    out = []
    for t in frame_times(start - timedelta(minutes=10), end + timedelta(minutes=10)):
        p = frame_path(root, t)
        if not p.exists():
            p = frame_path(root, t, partial=True)
            if not p.exists():
                continue
        for r in parse(p.read_text(encoding="utf-8")):
            key = (tuple(r[1:8]), r[8], r[9])
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


@dataclass(frozen=True)
class Strokes:
    """Located strokes, sorted by time: UTC time (datetime64[s]), position, and in-cloud flag (UALF cloud indicator)."""
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    in_cloud: np.ndarray  # bool

    def __len__(self) -> int:
        return self.time.size


def load_strokes(root: Path, start: datetime, end: datetime, radius_km: float = RADIUS_KM) -> Strokes:
    """Strokes with start <= time < end (UTC) within radius_km of the centre."""
    ts, la, lo, ic = [], [], [], []
    for r in read_rows(root, start, end):
        t = stroke_time_utc(r)
        lat, lon = float(r[8]), float(r[9])
        if start <= t < end and haversine_km(CENTER_LAT, CENTER_LON, lat, lon) <= radius_km:
            ts.append(t); la.append(lat); lo.append(lon)  # noqa: E702
            ic.append(len(r) > 21 and r[21] == "1")  # field 22 (CSV column c22): 1 = in-cloud, 0 = cloud-to-ground
    t = np.array(ts, dtype="datetime64[s]")
    order = np.argsort(t)
    return Strokes(t[order], np.array(la)[order], np.array(lo)[order], np.array(ic, bool)[order])


def export_csv(root: Path, start: datetime, end: datetime, out: Path, radius_km: float = RADIUS_KM) -> int:
    """The training CSV layout (time_utc, time_vn, lat, lon, dist_km, c11..)."""
    rows = []
    for r in read_rows(root, start, end):
        t = stroke_time_utc(r)
        d = haversine_km(CENTER_LAT, CENTER_LON, float(r[8]), float(r[9]))
        if start <= t < end and d <= radius_km:
            rows.append((t, [f"{t:%Y-%m-%d %H:%M:%S}", f"{t + VN:%Y-%m-%d %H:%M:%S}", r[8], r[9], f"{d:.1f}"] + r[10:]))
    rows.sort(key=lambda x: x[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEAD)
        w.writerows(r for _, r in rows)
    return len(rows)


def latency_record(root: Path, now: datetime, lookback_min: int) -> dict:
    """Newest stroke near the centre and strokes per 5-min frame over the lookback, as seen now."""
    s = load_strokes(root, now - timedelta(minutes=lookback_min), now + timedelta(minutes=5))
    newest = s.time.max() if len(s) else None
    bins = [floor5(now - timedelta(minutes=lookback_min)) + STEP * i for i in range(lookback_min // 5 + 2)]
    per = {f"{b:%H:%M}": int(((s.time >= np.datetime64(b)) & (s.time < np.datetime64(b + STEP))).sum()) for b in bins}
    return {"now_utc": f"{now:%Y-%m-%dT%H:%M:%S}", "newest_stroke_utc": str(newest) if newest is not None else None,
            "newest_age_min": round((np.datetime64(now, "s") - newest) / np.timedelta64(60, "s"), 1) if newest is not None else None,
            "strokes_per_frame": per}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "live", "backfill"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=Path("data/lightning/feed"))
        p.add_argument("--settle-min", type=float, default=20.0, help="Frames older than this are final")
        p.add_argument("--delay", type=float, default=0.3, help="Seconds between requests")
        if name in ("probe", "live"):
            p.add_argument("--lookback-min", type=int, default=40, help="Frames re-checked on each poll")
        if name == "live":
            p.add_argument("--poll-sec", type=float, default=60.0)
        if name == "backfill":
            p.add_argument("--start", type=datetime.fromisoformat, required=True, help="UTC")
            p.add_argument("--end", type=datetime.fromisoformat, required=True, help="UTC, exclusive")
            p.add_argument("--csv", type=Path, default=None, help="Also export the strokes in the training CSV layout")
    args = ap.parse_args()
    client = Client()
    if args.cmd == "backfill":
        sync(client, args.root, args.start, args.end, args.settle_min, args.delay)
        if args.csv:
            print(f"{export_csv(args.root, args.start, args.end, args.csv)} strokes -> {args.csv}")
        return
    seen: set[tuple] | None = None  # stroke keys seen so far; None until the first poll
    while True:
        now = utcnow()
        try:
            sync(client, args.root, now - timedelta(minutes=args.lookback_min), floor5(now) + STEP, args.settle_min, args.delay)
            seen_at = utcnow()
            rec = latency_record(args.root, seen_at, args.lookback_min)
            rows = read_rows(args.root, seen_at - timedelta(minutes=args.lookback_min), seen_at)
            keys = {(tuple(r[1:8]), r[8], r[9]): r for r in rows}
            if seen is not None:
                delays = np.array([(seen_at - stroke_time_utc(r)).total_seconds() / 60 for k, r in keys.items() if k not in seen])
                rec["new_strokes"] = int(delays.size)
                if delays.size:
                    rec.update({"delay_p50_min": round(float(np.median(delays)), 1),
                                "delay_p90_min": round(float(np.percentile(delays, 90)), 1),
                                "delay_max_min": round(float(delays.max()), 1)})
            seen = set(keys)  # older strokes leave the lookback and are never re-read
        except Exception as exc:  # noqa: BLE001  (keep polling through outages; they are logged)
            rec = {"now_utc": f"{now:%Y-%m-%dT%H:%M:%S}", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(rec), flush=True)
        if args.cmd == "probe":
            return
        with (args.root / "latency.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(max(0.0, args.poll_sec - (utcnow() - now).total_seconds()))


if __name__ == "__main__":
    main()
