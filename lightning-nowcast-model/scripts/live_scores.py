"""Scores the live forecasts whose windows are over against the iWeather strokes the feed archived.

    uv run python scripts/live_scores.py --work-dir live [--lightning-feed data/lightning/feed] [--day 2026-10-23]
                                         [--every-min 60]
    uv run python scripts/live_scores.py --work-dir shadow/2026-09-01_2026-10-23 --lightning-csv data/lightning/strokes_shadow.csv \
                                         [--first-anchor 2026-09-01T01:00]   (a replay: truth from a stroke CSV)

The levels' window (60 min, 10 km): a cell has the event when a stroke falls within 10 km of it in (T+10, T+60], a
commune when one of its cells has. Cells are the 2 km grid within 250 km; communes are those of
data/boundaries/communes_2025.geojson with cells there, valued as the daemon does (the highest probability over their
cells, recomputed from prob.npz, so forecasts written before the communes score the same way). Tables at the alert (30%) and warning
(40%) levels and for persistence (a stroke within 10 km in the hour up to T+5, what the model saw), the Brier score
against the day's base rate, reliability, and the share of strokes inside the >= 30% area (as the moment panels count).
Windows ending less than 10 min ago are left out (the feed's strokes arrive up to ~7 min late).

Prints the report and writes it to <work-dir>/scores/YYYYmmdd.txt; with --every-min, again every so often (one summary
line per run to stdout).
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import ndimage

from lightning_nowcast.app.geo import cell_of
from lightning_nowcast.grid import OUTER, RES_KM, grid_lonlat, range_km_map
from lightning_nowcast.lightning_feed import Strokes, load_strokes
from lightning_nowcast.pack import load_strokes as load_csv
from lightning_nowcast.product import Forecast
from lightning_nowcast.regions import commune_masks, communes

RADIUS_KM = 10.0
MAX_RANGE_KM = 250.0
SETTLE = timedelta(minutes=10)
VN = timedelta(hours=7)
LEVELS = (0.3, 0.4)
RELIABILITY = ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def cells(s: Strokes) -> tuple[np.ndarray, np.ndarray]:
    iy, ix = cell_of(s.lon, s.lat)
    ok = (iy >= 0) & (iy < OUTER) & (ix >= 0) & (ix < OUTER)
    return iy[ok], ix[ok]


def event(s: Strokes) -> np.ndarray:
    """Cells within RADIUS_KM of a stroke."""
    m = np.zeros((OUTER, OUTER), bool)
    m[cells(s)] = True
    r = int(RADIUS_KM / RES_KM)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_dilation(m, structure=(yy ** 2 + xx ** 2) <= (RADIUS_KM / RES_KM) ** 2)


def table(f: np.ndarray, o: np.ndarray) -> np.ndarray:
    """hits, misses, false alarms, correct negatives."""
    return np.array([(f & o).sum(), (~f & o).sum(), (f & ~o).sum(), (~f & ~o).sum()], np.int64)


def scores(t: np.ndarray) -> str:
    h, m, f, _ = t

    def div(a: float, b: float) -> str:
        return f"{a / b:.2f}" if b else "–"

    return f"POD {div(h, h + m)} FAR {div(f, h + f)} CSI {div(h, h + m + f)} bias {div(h + f, h + m)} (h/m/f {h}/{m}/{f})"


StrokeSource = Callable[[datetime, datetime], Strokes]  # strokes with start <= time < end


def from_feed(feed: Path) -> StrokeSource:
    return lambda t0, t1: load_strokes(feed, t0, t1)


def from_csv(path: Path) -> StrokeSource:
    df = load_csv(path).sort_values("t")
    t, lat, lon, cloud = df["t"].to_numpy(), df["lat"].to_numpy(), df["lon"].to_numpy(), df["cloud"].to_numpy() == 1

    def strokes(t0: datetime, t1: datetime) -> Strokes:
        i, j = np.searchsorted(t, [np.datetime64(t0, "s"), np.datetime64(t1, "s")])
        return Strokes(t[i:j], lat[i:j], lon[i:j], cloud[i:j])
    return strokes


def report(work_dir: Path, load: StrokeSource, day: str | None, first: datetime | None = None) -> tuple[str, str]:
    """The report and a one-line summary."""
    now = utcnow()
    in_range = range_km_map() <= MAX_RANGE_KM
    lon, lat = grid_lonlat()
    masks = list(commune_masks(communes(), lon, lat, in_range).values())
    cell_idx = np.concatenate([np.flatnonzero(m) for m in masks])  # every commune's cells, one after the other
    starts = np.cumsum([0] + [int(m.sum()) for m in masks])[:-1]

    def per_commune(field: np.ndarray) -> np.ndarray:
        """The highest value over each commune's cells."""
        return np.maximum.reduceat(field.ravel()[cell_idx].astype(np.float64), starts)

    keys = [f"cells {lv:.0%}" for lv in LEVELS] + ["cells persistence"] + [f"communes {lv:.0%}" for lv in LEVELS] + ["communes persistence"]
    total = {k: np.zeros(4, np.int64) for k in keys}
    probs, obs, rows = [], [], []
    strokes_in, strokes_all = 0, 0
    pattern = f"{day.replace('-', '')}/*/forecast.json" if day else "*/*/forecast.json"
    for fj in sorted((work_dir / "forecasts").glob(pattern)):
        fc = Forecast.read(fj.parent)
        k = fc.level_index
        t0, t1 = fc.anchor + timedelta(minutes=10), fc.anchor + timedelta(minutes=fc.windows[k].end_min)
        if t1 > now - SETTLE or (first is not None and fc.anchor < first):
            continue
        s = load(t0, t1)
        ob = event(s)
        pers = event(load(fc.anchor - timedelta(minutes=55), fc.anchor + timedelta(minutes=5)))
        p = fc.prob[k].astype(np.float64)
        cp, co = per_commune(p), per_commune(ob) > 0  # commune forecast / event
        for lv in LEVELS:
            total[f"cells {lv:.0%}"] += table(p[in_range] >= lv, ob[in_range])
            total[f"communes {lv:.0%}"] += table(cp >= lv, co)
        total["cells persistence"] += table(pers[in_range], ob[in_range])
        total["communes persistence"] += table(per_commune(pers) > 0, co)
        probs.append(p[in_range])
        obs.append(ob[in_range])
        iy, ix = cells(s)
        inside = int((p[iy, ix] >= LEVELS[0]).sum())
        strokes_in += inside
        strokes_all += iy.size
        t30 = table(p[in_range] >= LEVELS[0], ob[in_range])
        rows.append(f"| {fc.anchor + VN:%H:%M} | {iy.size} | {inside} | {int(ob[in_range].sum() * RES_KM ** 2)} | "
                    f"{int((p[in_range] >= LEVELS[0]).sum() * RES_KM ** 2)} | {t30[0]}/{t30[1]}/{t30[2]} | {float(p[in_range].max()):.2f} | "
                    f"{int(co.sum())}/{int((cp >= LEVELS[0]).sum())} |")
    head = f"Live scores at {now + VN:%H:%M %d/%m/%Y} (giờ VN): {len(rows)} forecasts with windows over"
    if not rows:
        return head + "\n", head
    lines = [head, "", "| T (VN) | strokes | in >= 30% | event area km² | warned >= 30% km² | cells h/m/f | max p | communes event/warned |",
             "|---|---|---|---|---|---|---|---|", *rows, "",
             f"Strokes inside the >= 30% area: {strokes_in}/{strokes_all}" + (f" ({strokes_in / strokes_all:.0%})" if strokes_all else ""),
             f"Cells (2 km, <= {MAX_RANGE_KM:g} km), event = stroke within {RADIUS_KM:g} km in T+10..T+60:"]
    lines += [f"  {k.split(' ', 1)[1]:>11s}  {scores(total[k])}" for k in keys[:3]]
    lines += [f"{len(masks)} communes:"] + [f"  {k.split(' ', 1)[1]:>11s}  {scores(total[k])}" for k in keys[3:]]
    P, O = np.concatenate(probs), np.concatenate(obs).astype(np.float64)
    base = O.mean()
    bs, bs_ref = float(np.mean((P - O) ** 2)), float(np.mean((base - O) ** 2))
    bss = 1 - bs / bs_ref if bs_ref else float("nan")
    lines.append(f"Brier {bs:.4f}, the day's base rate {bs_ref:.4f}, BSS {bss:+.2f} (base rate {base:.3f})")
    lines.append("Reliability (forecast probability -> observed frequency):")
    for lo, hi in RELIABILITY:
        m = (P >= lo) & (P < hi)
        if m.any():
            lines.append(f"  {lo:.1f}–{min(hi, 1):.1f}: {int(m.sum()):6d} cells, mean {P[m].mean():.2f} -> observed {O[m].mean():.2f}")
    c30 = total[keys[0]]
    summary = (f"{head}; cells >= 30%: {scores(c30)}; strokes inside {strokes_in}/{strokes_all}; BSS {bss:+.2f}")
    return "\n".join(lines) + "\n", summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--lightning-feed", type=Path, default=Path("data/lightning/feed"))
    ap.add_argument("--lightning-csv", type=Path, default=None, help="Truth from a stroke CSV (replays) instead of the feed")
    ap.add_argument("--first-anchor", type=datetime.fromisoformat, default=None, help="Leave out earlier forecasts (UTC)")
    ap.add_argument("--day", default=None, help="Only forecasts of this UTC day (YYYY-MM-DD); default: all")
    ap.add_argument("--every-min", type=float, default=None, help="Run again every so often (live)")
    args = ap.parse_args()
    load = from_csv(args.lightning_csv) if args.lightning_csv else from_feed(args.lightning_feed)
    while True:
        day = args.day or (f"{utcnow():%Y-%m-%d}" if args.every_min else None)
        text, summary = report(args.work_dir, load, day, args.first_anchor)
        out = args.work_dir / "scores" / f"{(day or 'all').replace('-', '')}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(summary if args.every_min else text, flush=True)
        if not args.every_min:
            break
        time.sleep(args.every_min * 60)


if __name__ == "__main__":
    main()
