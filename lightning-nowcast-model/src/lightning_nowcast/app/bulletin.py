"""The draft bulletin (Vietnamese, for the duty forecaster to check and edit) in the form of the Southern bulletins (the
wards and communes listed by province) and the GeoJSON of the warned communes."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

import numpy as np

from ..product import AreaWarning, Forecast, Level
from ..regions import Commune
from .geo import Geo
from .style import ICT, LEVEL_RANK, LEVEL_TEXT, SOURCE_NAMES, day, hhmm


def _by_province(names: list[tuple[str, str]]) -> list[str]:
    """'- Tỉnh Đồng Tháp: Xã A, Phường B.' lines, provinces in the order given (bulletin order)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for province, name in names:
        groups[province].append(name)
    return [f"- {province}: {', '.join(xs)}." for province, xs in groups.items()]


def inputs_line(fc: Forecast) -> str:
    missing = [f"{SOURCE_NAMES[s]} ({h.frames}/{h.needed})" for s, h in fc.health.items() if not h.complete]
    return "đầy đủ" if not missing else "thiếu " + ", ".join(missing)


def struck_communes(fc: Forecast, geo: Geo, minutes: float = 60.0) -> dict[str, int]:
    """Commune code -> strokes in it in the last `minutes` before T+5."""
    o = fc.observed
    recent = np.flatnonzero(o.stroke_min > 5 - minutes)
    out: dict[str, int] = defaultdict(int)
    for i in recent:
        c = geo.commune_at(float(o.stroke_lon[i]), float(o.stroke_lat[i]))
        if c is not None:
            out[c.code] += 1
    return dict(out)


def _listed(geo: Geo, areas: list[AreaWarning], k: int) -> list[tuple[str, str]]:
    """(province, name) of the communes, in bulletin order: provinces as geo lists them, the most likely first within."""
    rank = {c.code: i for i, c in enumerate(geo.communes)}
    first = {}
    for c in geo.communes:
        first.setdefault(c.province, rank[c.code])
    return [(a.province, a.name) for a in sorted(areas, key=lambda a: (first.get(a.province, 0), -a.probability[k]))]


def bulletin(fc: Forecast, geo: Geo) -> str:
    k = fc.level_index
    w = fc.windows[k]
    lv = fc.levels
    start, end = fc.anchor + timedelta(minutes=10), fc.anchor + timedelta(minutes=w.end_min)
    warned = [a for a in fc.communes if a.level == Level.WARNING]
    watched = [a for a in fc.communes if a.level == Level.ALERT]
    struck = struck_communes(fc, geo)
    now = [(c.province, c.name) for c in geo.communes if c.code in struck]
    lines = [
        "TIN CẢNH BÁO SÉT TRÊN KHU VỰC NAM BỘ – SẢN PHẨM AI (THỬ NGHIỆM)",
        f"Phát hành: {hhmm(fc.issued)} ngày {day(fc.issued)} (giờ Việt Nam). Radar Nhà Bè lúc {hhmm(fc.anchor)}.",
        "",
        f"1. Hiện trạng trong 60 phút qua (đến {hhmm(fc.anchor + timedelta(minutes=5))}): định vị sét ghi nhận sét trên các phường/xã:",
        *(_by_province(now) or ["   Không ghi nhận sét trên các phường/xã của khu vực."]),
        f"2. Cảnh báo khả năng xuất hiện sét từ {hhmm(start)} đến {hhmm(end)} ngày {day(end)}, trong bán kính {w.radius_km:g} km:",
        f"a) {LEVEL_TEXT[Level.WARNING]} (xác suất từ {lv.warning:.0%}) – {len(warned)} phường/xã:",
        *(_by_province(_listed(geo, warned, k)) or ["   Không có."]),
        f"b) {LEVEL_TEXT[Level.ALERT]} (xác suất từ {lv.alert:.0%} đến dưới {lv.warning:.0%}) – {len(watched)} phường/xã:",
        *(_by_province(_listed(geo, watched, k)) or ["   Không có."]),
    ]
    longest = len(fc.windows) - 1
    if longest != k:
        lw = fc.windows[longest]
        later = [a for a in fc.communes if a.level == Level.NONE and a.probability[longest] >= lv.warning]
        by = defaultdict(int)
        for a in later:
            by[a.province] += 1
        lines.append(f"3. Đến {hhmm(fc.anchor + timedelta(minutes=lw.end_min))} (bán kính {lw.radius_km:g} km), khả năng có sét mở rộng sang "
                     + (", ".join(f"{n} phường/xã của {p}" for p, n in by.items()) if by else "không có phường/xã mới") + ".")
    lines += ["", f"Mô hình: {fc.model}. Dữ liệu đầu vào: {inputs_line(fc)}.",
              f"Xác suất là khả năng có ít nhất một cú sét trong bán kính {w.radius_km:g} km quanh phường/xã trong thời gian hiệu lực."]
    return "\n".join(lines)


def geojson(fc: Forecast, geo: Geo, min_level: Level = Level.ALERT) -> dict[str, Any]:
    """FeatureCollection of the communes at `min_level` or above: each its own outline, with its probabilities and the
    model version."""
    k = fc.level_index
    ict = lambda t: (t + ICT).replace(microsecond=0).isoformat() + "+07:00"  # noqa: E731
    names = [f"{w.end_min}r{w.radius_km:g}" for w in fc.windows]
    feats = []
    for a in fc.communes:
        c: Commune | None = geo.by_code.get(a.code)
        if LEVEL_RANK[a.level] < LEVEL_RANK[min_level] or c is None:
            continue
        rings = [[[round(float(x), 5), round(float(y), 5)] for x, y in r] for r in c.rings]
        feats.append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in rings]},
                      "properties": {"code": a.code, "name": a.name, "province": a.province,
                                     "level": a.level.value, "probability": a.probability[k],
                                     "probabilities": dict(zip(names, a.probability)),
                                     "valid_from": ict(fc.anchor + timedelta(minutes=10)),
                                     "valid_to": ict(fc.anchor + timedelta(minutes=fc.windows[k].end_min)),
                                     "issued": ict(fc.issued), "model_version": fc.model,
                                     "inputs_complete": all(h.complete for h in fc.health.values())}})
    return {"type": "FeatureCollection", "features": feats}
