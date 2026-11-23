"""Compare the official thunderstorm bulletins (Đài KTTV Nam Bộ, .docx) with the AI forecasts of a replay, commune by
commune, against the same lightning truth.

    uv run --with python-docx python scripts/compare_bulletins.py --work-dir shadow/2026-09-01_2026-10-23 \
        --bulletins ../quy-trinh/ban-tin-T9-T10 --lightning-csv data/lightning/strokes_shadow.csv

What a bulletin warns: the communes it names (the storms of the last 0-3 h, warned for "trong 0-3 giờ tới" together with
"các khu vực lân cận", which is not a list and so cannot be scored). Named in a dossier (HS_*) in the rows
"Phạm vi xảy ra ..." and "Kết luận ..."; in a bulletin (NABO_*) under "1. Hiện trạng". HOCM bulletins cover Hồ Chí Minh
only, so they are scored on its communes; NABO on all nine provinces. Only communes with cells within 250 km.

Against, with T the bulletin's time:
  - AI 30% / 40%: the communes at or above that level in the latest forecast issued by T (60-min window);
  - AI 40% updated: the communes warned at 40% by any forecast issued in [T, T+2 h] (what an operator following the
    product every 10 min would have warned over the bulletin's 3 h);
  - persistence: the communes struck in the last 1 h / 3 h before T (roughly what the bulletin itself does).
Truth: a stroke in the commune (or within 10 km of it) in (T, T+1 h] and (T, T+3 h].
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from live_scores import MAX_RANGE_KM, cells, event, from_csv, scores, table  # noqa: E402

from lightning_nowcast.grid import OUTER, grid_lonlat, range_km_map  # noqa: E402
from lightning_nowcast.product import Forecast  # noqa: E402
from lightning_nowcast.regions import commune_masks, communes  # noqa: E402

VN = timedelta(hours=7)
TONE = {"òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ", "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ"}
UNIT = re.compile(r"(?:phường|xã|đặc khu) [^,()\.;:\n]+")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    for old, new in TONE.items():
        s = s.replace(old, new)
    return re.sub(r"\s+", " ", s)


@dataclass
class Bulletin:
    name: str
    time: datetime  # UTC
    scope: str  # "HOCM" or "NABO"
    codes: set[str]
    unmatched: list[str]


def text_of(path: Path) -> tuple[str, str]:
    import docx
    d = docx.Document(str(path))
    if path.name.startswith("HS_"):
        parts = []
        for t in d.tables:
            for r in t.rows:
                cs = list(dict.fromkeys(c.text for c in r.cells))
                if cs and (cs[0].startswith("Phạm vi xảy ra") or cs[0].startswith("Kết luận (thời gian")):
                    parts += cs[1:]
        return "\n".join(parts), ("HOCM" if "_HOCM_" in path.name else "NABO")
    ps = [p.text for p in d.paragraphs]
    i = next(i for i, p in enumerate(ps) if p.startswith("1. Hiện trạng"))
    j = next((j for j in range(i, len(ps)) if ps[j].startswith("2. Cảnh báo")), len(ps))
    return "\n".join(ps[i:j]), ("HOCM" if "_HOCM_" in path.name else "NABO")


def parse(path: Path, units: list) -> Bulletin:
    text, scope = text_of(path)
    t = norm(text)
    provinces = sorted({u.province for u in units})
    # where each province's part starts (the bulletins group the communes by province)
    # (bulletins write "Tỉnh Đồng Nai" or "Thành phố Đồng Nai"; match on the name after the prefix)
    short = {p: re.sub(r"^(tỉnh|thành phố) ", "", norm(p)) for p in provinces}
    marks = sorted((m.start(), p) for p in provinces for m in re.finditer(r"(?:tỉnh|thành phố|tp\.?) " + re.escape(short[p]), t))
    default = "Thành phố Hồ Chí Minh" if scope == "HOCM" else None
    names: dict[str, dict[str, str]] = {}
    for u in units:
        names.setdefault(u.province, {})[norm(u.name)] = u.code
    for u in units:  # the boundaries are newer than some bulletins: Xã X may since have become Phường X, or back
        n = norm(u.name)
        for a, b in (("phường ", "xã "), ("xã ", "phường ")):
            if n.startswith(a):
                names[u.province].setdefault(b + n[len(a):], u.code)
    codes, spans = set(), []
    for p, table_ in names.items():
        pat = re.compile("(" + "|".join(re.escape(n) for n in sorted(table_, key=len, reverse=True)) + r")(?![\w])")
        for m in pat.finditer(t):
            prov = next((q for s, q in reversed(marks) if s <= m.start()), default)
            if prov == p:
                codes.add(table_[m.group(1)])
                spans.append((m.start(), m.end()))
    unmatched = [m.group(0).strip() for m in UNIT.finditer(t)
                 if not any(s <= m.start() < e for s, e in spans)]
    stamp = re.search(r"(\d{8})_(\d{4})", path.name)
    return Bulletin(path.stem, datetime.strptime("".join(stamp.groups()), "%Y%m%d%H%M") - VN, scope, codes,
                    sorted(set(unmatched)))


def latest_issued_by(work: Path, t: datetime) -> Forecast | None:
    for back in range(0, 40, 10):
        a = t - timedelta(minutes=back)
        a = a.replace(minute=a.minute // 10 * 10, second=0, microsecond=0)
        d = work / "forecasts" / f"{a:%Y%m%d}" / f"{a:%H%M}"
        if (d / "forecast.json").exists():
            fc = Forecast.read(d)
            if fc.issued <= t:
                return fc
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--bulletins", type=Path, default=Path("../quy-trinh/ban-tin-T9-T10"))
    ap.add_argument("--lightning-csv", type=Path, default=Path("data/lightning/strokes_shadow.csv"))
    args = ap.parse_args()

    units = communes()
    in_range = range_km_map() <= MAX_RANGE_KM
    lon, lat = grid_lonlat()
    masks = commune_masks(units, lon, lat, in_range)
    codes = list(masks)
    by_code = {u.code: u for u in units}
    cell_idx = np.concatenate([np.flatnonzero(masks[c]) for c in codes])
    starts = np.cumsum([0] + [int(masks[c].sum()) for c in codes])[:-1]

    def per_commune(field: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(field.ravel()[cell_idx].astype(np.float64), starts)

    load = from_csv(args.lightning_csv)

    def struck(t0: datetime, t1: datetime, near: bool) -> np.ndarray:
        s = load(t0, t1)
        if near:
            return per_commune(event(s)) > 0
        m = np.zeros((OUTER, OUTER), bool)
        m[cells(s)] = True
        return per_commune(m) > 0

    def warned(fc: Forecast | None, lv: float, end_min: int | None = None) -> np.ndarray:
        """Communes at or above `lv` in the levels' window, or in the window ending at T+end_min."""
        if fc is None:
            return np.zeros(len(codes), bool)
        k = fc.level_index if end_min is None else next(i for i, w in enumerate(fc.windows) if w.end_min == end_min)
        return per_commune(fc.prob[k]) >= lv

    order = ["Bản tin", "AI 30%", "AI 40%", "AI 120' 40%", "AI 120' 60%", "AI 40% cập nhật", "Persistence 1 h", "Persistence 3 h"]
    truths = ["1 h trong xã", "3 h trong xã", "1 h ≤10 km", "3 h ≤10 km"]
    tot = {(s, h, o): np.zeros(4, np.int64) for s in ["HOCM", "NABO"] for h in truths for o in order}
    rows, skipped = [], []
    for path in sorted(args.bulletins.glob("*DONG_2026*.docx")):
        if path.name.startswith("HOCM_DONG_"):
            continue  # the August ones without a dossier: before the replay
        b = parse(path, units)
        fc = latest_issued_by(args.work_dir, b.time)
        later = [latest_issued_by(args.work_dir, b.time + timedelta(minutes=m)) for m in range(10, 121, 10)]
        if fc is None or any(f is None for f in later):
            skipped.append(f"{b.name} (no replay yet)")
            continue
        dom = np.array([b.scope == "NABO" or by_code[c].province_code == "79" for c in codes])
        bul = np.isin(codes, list(b.codes))
        upd = np.logical_or.reduce([warned(f, 0.4) for f in [fc, *later]])
        sets = {"Bản tin": bul, "AI 30%": warned(fc, 0.3), "AI 40%": warned(fc, 0.4),
                "AI 120' 40%": warned(fc, 0.4, 120), "AI 120' 60%": warned(fc, 0.6, 120), "AI 40% cập nhật": upd,
                "Persistence 1 h": struck(b.time - timedelta(hours=1), b.time, False),
                "Persistence 3 h": struck(b.time - timedelta(hours=3), b.time, False)}
        obs = {"1 h trong xã": struck(b.time, b.time + timedelta(hours=1), False),
               "3 h trong xã": struck(b.time, b.time + timedelta(hours=3), False),
               "1 h ≤10 km": struck(b.time, b.time + timedelta(hours=1), True),
               "3 h ≤10 km": struck(b.time, b.time + timedelta(hours=3), True)}
        for h, o in obs.items():
            for name, f in sets.items():
                tot[(b.scope, h, name)] += table(f[dom], o[dom])
        t3 = table(bul[dom], obs["3 h trong xã"][dom])
        a3 = table(upd[dom], obs["3 h trong xã"][dom])
        rows.append(f"| {b.name} | {b.time + VN:%d/%m %H:%M} | {int(dom.sum())} | {int(bul[dom].sum())} | "
                    f"{int(sets['AI 40%'][dom].sum())} | {int(upd[dom].sum())} | {int(obs['3 h trong xã'][dom].sum())} | "
                    f"{t3[0]}/{t3[1]}/{t3[2]} | {a3[0]}/{a3[1]}/{a3[2]} | {len(b.unmatched)} |")
        if b.unmatched:
            rows.append(f"|  unmatched: {'; '.join(b.unmatched[:12])} |")

    out = ["# Bản tin chính thức và dự báo AI, theo phường/xã", "",
           "| Bản tin | giờ VN | phường/xã | bản tin nêu | AI 40% | AI 40% cập nhật 2 h | có sét 3 h (trong xã) | bản tin h/m/f 3 h | AI cập nhật h/m/f 3 h | tên không khớp |",
           "|---|---|---|---|---|---|---|---|---|---|", *rows, ""]
    if skipped:
        out.append("Not yet scored: " + ", ".join(skipped))
    for s in ["HOCM", "NABO"]:
        if not any(tot[(s, truths[0], o)].sum() for o in order):
            continue
        out += ["", f"## {s} ({'Hồ Chí Minh' if s == 'HOCM' else 'Nam Bộ'})"]
        for h in truths:
            out.append(f"Truth: stroke {h}")
            out += [f"  {o:>17s}  {scores(tot[(s, h, o)])}" for o in order]
    text = "\n".join(out) + "\n"
    dst = args.work_dir / "scores" / "bulletins.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
