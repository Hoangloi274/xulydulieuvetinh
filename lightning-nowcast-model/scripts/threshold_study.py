"""How the warning level (now 40%) trades detections, false alarms and warning lead, commune by commune, over a replay.

    uv run python scripts/threshold_study.py --work-dir shadow/2026-09-01_2026-10-23 \
        --lightning-csv data/lightning/strokes_shadow.csv [--first-anchor 2026-09-01T01:00]

Truth on 10-min bins: a commune has lightning in a bin when a stroke falls within 10 km of it (as live_scores.py, on
10-min boundaries). For each forecast (anchor A) the commune value is the highest 60-min probability over its cells;
the event is lightning in any bin of [A+10, A+60).

1. POD / FAR / CSI / bias of "warned = value >= level" for levels 0.30 .. 0.80, summed over all forecasts.
2. Reliability: forecast value bins -> observed frequency.
3. Onsets: a commune's first lightning bin after at least 60 min without. Of the forecasts issued before it whose
   window covers it (anchors A with A+10 <= onset < A+60), was any at or above the level, and how long before the
   onset was the earliest such forecast issued (issue = A+9 min; lead counted to the start of the onset bin, so it is
   a lower bound by up to 10 min)?
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from live_scores import MAX_RANGE_KM, event, from_csv  # noqa: E402

from lightning_nowcast.grid import grid_lonlat, range_km_map  # noqa: E402
from lightning_nowcast.product import Forecast  # noqa: E402
from lightning_nowcast.regions import commune_masks, communes  # noqa: E402

BIN = timedelta(minutes=10)
LEVELS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
ISSUE = 9  # minutes after the anchor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--lightning-csv", type=Path, required=True)
    ap.add_argument("--first-anchor", type=datetime.fromisoformat, default=datetime(2026, 9, 1, 1))
    args = ap.parse_args()

    in_range = range_km_map() <= MAX_RANGE_KM
    lon, lat = grid_lonlat()
    masks = commune_masks(communes(), lon, lat, in_range)
    codes = list(masks)
    cell_idx = np.concatenate([np.flatnonzero(masks[c]) for c in codes])
    starts = np.cumsum([0] + [int(masks[c].sum()) for c in codes])[:-1]

    def per_commune(field: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(field.ravel()[cell_idx].astype(np.float32), starts)

    folders = sorted(p.parent for p in (args.work_dir / "forecasts").glob("*/*/forecast.json"))
    anchors, values = [], []
    for f in folders:
        fc = Forecast.read(f)
        if fc.anchor < args.first_anchor:
            continue
        anchors.append(fc.anchor)
        values.append(per_commune(fc.prob[fc.level_index]))
    V = np.array(values, np.float32)  # forecasts x communes
    t0 = min(anchors) - timedelta(hours=1)
    nbins = int((max(anchors) + timedelta(minutes=70) - t0) / BIN) + 1
    load = from_csv(args.lightning_csv)
    E = np.zeros((nbins, len(codes)), bool)  # bins x communes
    for b in range(nbins):
        s = load(t0 + b * BIN, t0 + (b + 1) * BIN)
        if s.lat.size:
            E[b] = per_commune(event(s)) > 0
    print(f"{len(anchors)} forecasts, {len(codes)} communes, {nbins} bins")

    bidx = np.array([int((a - t0) / BIN) for a in anchors])
    O = np.zeros_like(V, bool)
    for k in range(1, 6):  # bins A+10 .. A+50
        O |= E[bidx + k]

    print("\n1. Commune scores (stroke within 10 km, 60-min window)")
    print("level   POD   FAR   CSI  bias   warned/forecast")
    for lv in LEVELS:
        F = V >= lv
        h, m, f = int((F & O).sum()), int((~F & O).sum()), int((F & ~O).sum())
        print(f" {lv:.2f}  {h/(h+m):.2f}  {f/(h+f):.2f}  {h/(h+m+f):.2f}  {(h+f)/(h+m):.2f}   {F.sum()/len(V):6.1f}")

    print("\n2. Reliability (commune value -> observed frequency)")
    for lo, hi in [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        m = (V >= lo) & (V < hi)
        print(f" {lo:.1f}-{min(hi, 1):.1f}: {int(m.sum()):8d}  mean {V[m].mean():.2f} -> observed {O[m].mean():.2f}")

    # onsets: first lightning bin after >= 6 empty bins
    prev = np.zeros_like(E)
    for k in range(1, 7):
        prev[k:] |= E[:-k]
    onset = E & ~prev
    onset[:7] = False
    a_of_bin = {int(b): i for i, b in enumerate(bidx)}
    ob, oc = np.nonzero(onset)
    keep = []
    for b, c in zip(ob, oc):
        cand = [a_of_bin.get(b - k) for k in range(5, 0, -1)]  # anchors A = onset-50 .. onset-10 min, earliest first
        if all(x is not None for x in cand):
            keep.append((b, c, cand))
    print(f"\n3. Onsets with all 5 covering forecasts available: {len(keep)}")
    print("level  warned before onset  median lead (min)  lead >= 30 min")
    for lv in LEVELS:
        leads = []
        for b, c, cand in keep:
            for j, i in enumerate(cand):  # j=0 is anchor onset-50
                if V[i, c] >= lv:
                    leads.append((5 - j) * 10 - ISSUE)
                    break
        n = len(keep)
        leads = np.array(leads)
        print(f" {lv:.2f}  {len(leads)/n:6.0%}              {np.median(leads) if leads.size else float('nan'):5.0f}"
              f"             {(leads >= 30).sum()/n:5.0%}")


if __name__ == "__main__":
    main()
