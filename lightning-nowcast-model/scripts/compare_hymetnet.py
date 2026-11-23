"""Compare hymetnet's thunderstorm warnings (hymetnet.gov.vn/dongset, archived every 5 min during the live runs) with
our live forecasts, commune by commune, against the same lightning truth.

    uv run python scripts/compare_hymetnet.py [--work-dir live] [--lightning-csv data/lightning/strokes_shadow.csv]

/dongset lists, per 10-min valid time, the communes ("Xã X, Tỉnh Y, lat-lon") hymetnet warns; a page fetched at F
holds the valid times F-20 .. F+30. For each of our forecasts (anchor A, issued ~A+9) we take the latest /dongset fetched
by our issue time and call warned the communes it lists for any valid time in (A+10, A+60] (in practice up to ~A+35:
its horizon is shorter than ours). Ours: the communes at >= 30% / >= 40% (60-min window). Truth: a stroke in the
commune / within 10 km of it in (A+10, A+60], and in (A+10, A+40] (hymetnet's horizon). The nine provinces' communes
with cells within 250 km; hymetnet's other provinces are left out.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_bulletins import norm  # noqa: E402
from live_scores import MAX_RANGE_KM, cells, event, from_csv, from_feed, scores, table  # noqa: E402

from lightning_nowcast.grid import OUTER, grid_lonlat, range_km_map  # noqa: E402
from lightning_nowcast.product import Forecast  # noqa: E402
from lightning_nowcast.regions import commune_masks, communes  # noqa: E402

VN = timedelta(hours=7)


def short(p: str) -> str:
    return re.sub(r"^(tỉnh|thành phố) ", "", norm(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, default=Path("live"))
    ap.add_argument("--lightning-csv", type=Path, default=Path("data/lightning/strokes_shadow.csv"))
    ap.add_argument("--lightning-feed", type=Path, default=Path("data/lightning/feed"))
    ap.add_argument("--forecasts-from", type=Path, nargs="*", default=[],
                    help="Work dirs whose forecasts replace those of --work-dir for the same anchors (corrected replays)")
    args = ap.parse_args()

    units = communes()
    in_range = range_km_map() <= MAX_RANGE_KM
    lon, lat = grid_lonlat()
    masks = commune_masks(units, lon, lat, in_range)
    codes = list(masks)
    pos = {c: i for i, c in enumerate(codes)}
    lookup = {}
    for u in units:
        if u.code not in pos:
            continue
        n = norm(u.name)
        lookup[(short(u.province), n)] = u.code
    for u in units:  # Xã X <-> Phường X (the boundaries may be newer)
        if u.code in pos:
            n = norm(u.name)
            for a, b in (("phường ", "xã "), ("xã ", "phường ")):
                if n.startswith(a):
                    lookup.setdefault((short(u.province), b + n[len(a):]), u.code)
    cell_idx = np.concatenate([np.flatnonzero(masks[c]) for c in codes])
    starts = np.cumsum([0] + [int(masks[c].sum()) for c in codes])[:-1]

    def per_commune(field: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(field.ravel()[cell_idx].astype(np.float64), starts)

    csv, feed = from_csv(args.lightning_csv), from_feed(args.lightning_feed)
    csv_end = datetime(2026, 10, 23, 12)

    def struck(t0: datetime, t1: datetime, near: bool) -> np.ndarray:
        s = (csv if t1 <= csv_end else feed)(t0, t1)
        if near:
            return per_commune(event(s)) > 0
        m = np.zeros((OUTER, OUTER), bool)
        m[cells(s)] = True
        return per_commune(m) > 0

    pages = []  # (fetch time UTC, path)
    for p in sorted((args.work_dir / "hymetnet").glob("*/*_dongset.json")):
        pages.append((datetime.strptime(p.parent.name + p.name[:4], "%Y%m%d%H%M"), p))
    unknown: set[str] = set()
    names = ["hymetnet", "AI 30%", "AI 40%"]
    truths = {"60 phút, trong xã": (60, False), "60 phút, ≤10 km": (60, True),  # window end (min after A), radius
              "30 phút, trong xã": (40, False), "30 phút, ≤10 km": (40, True)}
    provinces = {short(u.province) for u in units}
    tot = {(d, t, n): np.zeros(4, np.int64) for d in ("all", "20261023", "20261104") for t in truths for n in names}
    rows = []
    folders = {fj.parent.relative_to(d / "forecasts"): fj.parent
               for d in [args.work_dir, *args.forecasts_from] for fj in (d / "forecasts").glob("*/*/forecast.json")}
    for key in sorted(folders):
        fc = Forecast.read(folders[key])
        A = fc.anchor
        got = [(f, p) for f, p in pages if f <= fc.issued and f >= fc.issued - timedelta(minutes=15)]
        if not got:
            continue
        data = json.loads(got[-1][1].read_text(encoding="utf-8"))
        hm = np.zeros(len(codes), bool)
        for key, items in data.items():
            v = datetime.strptime(key, "%Y%m%d%H%M")
            if not (A + timedelta(minutes=10) < v <= A + timedelta(minutes=60)):
                continue
            for it in items:
                name, prov, _ = (x.strip() for x in it.rsplit(",", 2))
                code = lookup.get((short(prov), norm(name)))
                if code is not None:
                    hm[pos[code]] = True
                elif short(prov) in provinces:
                    unknown.add(f"{name}, {prov}")
        p = per_commune(fc.prob[fc.level_index])
        sets = {"hymetnet": hm, "AI 30%": p >= 0.3, "AI 40%": p >= 0.4}
        obs = {t: struck(A + timedelta(minutes=10), A + timedelta(minutes=m), near)
               for t, (m, near) in truths.items()}
        day = f"{A:%Y%m%d}"
        for t, o in obs.items():
            for n, f in sets.items():
                for d in ("all", day):
                    tot[(d, t, n)] += table(f, o)
        rows.append(f"| {A + VN:%d/%m %H:%M} | {got[-1][0] + VN:%H:%M} | {int(hm.sum())} | {int(sets['AI 40%'].sum())} | "
                    f"{int(obs['60 phút, ≤10 km'].sum())} |")
    out = ["# hymetnet /dongset và dự báo AI (chạy trực tiếp), theo phường/xã", "",
           "| T (VN) | dongset lúc | hymetnet nêu | AI 40% | có sét ≤10 km, 60 phút |", "|---|---|---|---|---|", *rows, ""]
    for d in ("all", "20261023", "20261104"):
        out.append(f"## {d}")
        for t in truths:
            out.append(f"Truth: {t}")
            out += [f"  {n:>9s}  {scores(tot[(d, t, n)])}" for n in names]
    if unknown:
        out.append("Not matched: " + "; ".join(sorted(unknown)))
    text = "\n".join(out) + "\n"
    dst = args.work_dir / "scores" / "hymetnet.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
