"""Map one official bulletin against the AI warnings over the same 3 h: each commune a hit, miss or false alarm.

    uv run --with python-docx python scripts/bulletin_map.py --bulletin ../quy-trinh/ban-tin-T9-T10/NABO_DONG_20260927_1410.docx \
        --work-dir shadow/2026-09-01_2026-10-23 --out map.png [--near]

Left: the communes the bulletin names. Right: the communes warned (>= 40%) by any AI forecast issued from the bulletin's
time to 2 h after (the product followed every 10 min). Truth: a stroke in the commune (--near: within 10 km) within
3 h of the bulletin. As scripts/compare_bulletins.py scores them.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from compare_bulletins import VN, latest_issued_by, parse  # noqa: E402
from live_scores import MAX_RANGE_KM, cells, event, from_csv  # noqa: E402

from lightning_nowcast.app.geo import Geo  # noqa: E402
from lightning_nowcast.grid import OUTER, grid_lonlat, range_km_map  # noqa: E402
from lightning_nowcast.regions import (  # noqa: E402
    DEFAULT_BOUNDARIES,
    DEFAULT_NOTABLE,
    commune_masks,
    communes,
)

HIT, MISS, FALSE, NONE = "#00897b", "#d9480f", "#9b6fc0", "#e7e3d8"
INK, MUTED = "#26231f", "#6b665d"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bulletin", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--lightning-csv", type=Path, default=Path("data/lightning/strokes_shadow.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--near", action="store_true", help="Truth: a stroke within 10 km (default: in the commune)")
    args = ap.parse_args()

    units = communes()
    in_range = range_km_map() <= MAX_RANGE_KM
    lon, lat = grid_lonlat()
    masks = commune_masks(units, lon, lat, in_range)
    codes = list(masks)
    b = parse(args.bulletin, units)
    t = b.time
    fcs = [latest_issued_by(args.work_dir, t + timedelta(minutes=m)) for m in range(0, 121, 10)]
    s = from_csv(args.lightning_csv)(t, t + timedelta(hours=3))
    if args.near:
        field = event(s)
    else:
        field = np.zeros((OUTER, OUTER), bool)
        field[cells(s)] = True
    struck = {c for c in codes if field[masks[c]].any()}
    named = b.codes & set(codes)
    ai = {c for c in codes if any(float(f.prob[f.level_index][masks[c]].max()) >= 0.4 for f in fcs)}

    geo = Geo.load(DEFAULT_BOUNDARIES, DEFAULT_NOTABLE)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=250)
    fig.patch.set_facecolor("white")
    titles = [f"Bản tin {t + VN:%H:%M}: nêu {len(named)} phường, xã",
              f"AI ≥ 40%, {fcs[0].issued + VN:%H:%M}–{fcs[-1].issued + VN:%H:%M}: {len(ai)} phường, xã"]
    for ax, warned, title in zip(axes, (named, ai), titles):
        polys, colors = [], []
        h = m = f = 0
        for c in geo.communes:
            if c.code not in masks:
                continue
            w, o = c.code in warned, c.code in struck
            col = HIT if w and o else MISS if o else FALSE if w else NONE
            h += w and o
            m += o and not w
            f += w and not o
            for r in c.rings:
                polys.append(np.asarray(r))
                colors.append(col)
        ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="white", linewidths=0.15))
        ax.set_xlim(104.75, 108.75)
        ax.set_ylim(8.55, 12.35)
        ax.set_aspect(1 / np.cos(np.radians(10.5)))
        ax.axis("off")
        ax.set_title(title, fontsize=9, color=INK, loc="left", pad=2)
        ax.text(0.0, 0.0, f"Đúng {h} · Bỏ sót {m}\nBáo khống {f} · CSI {h / max(h + m + f, 1):.2f}".replace(".", ","),
                transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")
    fig.legend(handles=[Patch(color=HIT, label="Đúng: có cảnh báo và có sét"), Patch(color=MISS, label="Bỏ sót: có sét, không cảnh báo"),
                        Patch(color=FALSE, label="Báo khống: có cảnh báo, không có sét"), Patch(color=NONE, label="Không cảnh báo, không có sét")],
               loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.13, wspace=0.02)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"{args.out}: struck {len(struck)}, named {len(named)}, ai {len(ai)}")


if __name__ == "__main__":
    main()
