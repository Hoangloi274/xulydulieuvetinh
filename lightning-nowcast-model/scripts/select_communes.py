# /// script
# requires-python = ">=3.11"
# dependencies = ["pyshp", "shapely>=2"]
# ///
"""Build the commune boundaries the forecast is issued for, and choose the notable ones: those named on the operator
map and listed first (run again only to change them). The forecast covers every commune; the notable list only decides
what is labelled and shown by default.

    uv run scripts/select_communes.py [--geojson geojson] [--out data/communes_notable.csv]
        [--boundaries data/boundaries/communes_2025.geojson]

Writes data/boundaries/communes_2025.geojson: every commune of the nine provinces, simplified to ~30 m, with its code,
full name, province (full name, as the bulletins write it: "Tỉnh Đồng Tháp", "Thành phố Hồ Chí Minh") and a label
point inside it.

Communes are the 2025 units (after the 1 July 2025 reform) of geojson/<province>/wards/*.geojson, the official
commune layer (diaphanhanhchinhcapxa_2025). Only the nine provinces of the Southern bulletins and only communes within
250 km of the radar are considered. Fixed criteria, in order:

1. centre: the ward that carries the name of each former city or town (GADM 4.1 level 2 of type City / Town, before
   the reform), plus the cores of the two centrally run cities (Phường Sài Gòn, Phường Ninh Kiều). Where the reform
   renamed the central ward, the ward that contains the old centre (RENAMED); those not yet confirmed are marked
   in the `check` column.
2. city: fixed extra wards of large cities (CITY).
3. spread: the rest of each province's quota, by farthest-point sampling: the commune farthest from every one already
   chosen, distances scaled by max(3 km, 0.18 x range) so the set stays denser near the radar, where people live and
   the forecast is best. Islands are left out of this step (only Côn Đảo is kept, below).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import unicodedata
from pathlib import Path

import shapefile
from shapely.geometry import Point, shape

RADAR = (106.7283, 10.6595)  # Nhà Bè (grid.CENTER_LON, CENTER_LAT)
MAX_RANGE_KM = 250.0
GADM_DISTRICTS = Path("data/boundaries/gadm41_VNM_2.shp")
# province (2025) -> (folder in the commune layer, quota)
QUOTA = {"Hồ Chí Minh": ("79_ho_chi_minh", 20), "Đồng Nai": ("75_dong_nai", 12), "Tây Ninh": ("80_tay_ninh", 12),
         "Đồng Tháp": ("82_dong_thap", 12), "Vĩnh Long": ("86_vinh_long", 12), "An Giang": ("91_an_giang", 12),
         "Cần Thơ": ("92_can_tho", 12), "Lâm Đồng": ("68_lam_dong", 8), "Cà Mau": ("96_ca_mau", 6)}
# former province (GADM NAME_1) -> province after the reform
MERGED = {"Hồ Chí Minh": "Hồ Chí Minh", "Bình Dương": "Hồ Chí Minh", "Bà Rịa - Vũng Tàu": "Hồ Chí Minh", "Đồng Nai": "Đồng Nai",
          "Bình Phước": "Đồng Nai", "Tây Ninh": "Tây Ninh", "Long An": "Tây Ninh", "Đồng Tháp": "Đồng Tháp", "Tiền Giang": "Đồng Tháp",
          "Vĩnh Long": "Vĩnh Long", "Bến Tre": "Vĩnh Long", "Trà Vinh": "Vĩnh Long", "An Giang": "An Giang", "Kiên Giang": "An Giang",
          "Cần Thơ": "Cần Thơ", "Hậu Giang": "Cần Thơ", "Sóc Trăng": "Cần Thơ", "Cà Mau": "Cà Mau", "Bạc Liêu": "Cà Mau",
          "Lâm Đồng": "Lâm Đồng", "Đắk Nông": "Lâm Đồng", "Bình Thuận": "Lâm Đồng"}
CORES = [("Hồ Chí Minh", "Sài Gòn"), ("Cần Thơ", "Ninh Kiều")]
# former cities whose central ward was renamed: the old centre (lon, lat); True = to be confirmed by the operators
RENAMED = {("Cà Mau", "Cà Mau"): (105.1907, 9.1718, True), ("Vĩnh Long", "Vĩnh Long"): (105.9370, 10.2526, False),
           ("Tây Ninh", "Tây Ninh"): (106.1267, 11.3688, True), ("Lâm Đồng", "Bảo Lộc"): (107.8070, 11.5480, False),
           ("Lâm Đồng", "Gia Nghĩa"): (107.6900, 12.0040, False), ("Lâm Đồng", "Đà Lạt"): (108.4580, 11.9400, False)}
CITY = [("Hồ Chí Minh", "Phường Thủ Đức"), ("Hồ Chí Minh", "Phường Bình Tân"), ("Hồ Chí Minh", "Đặc khu Côn Đảo")]
TONE = {"òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ", "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ"}


def norm(s: str) -> str:
    """Names compare equal whatever the tone-mark placement (Hòa / Hoà)."""
    s = unicodedata.normalize("NFC", s).lower().strip()
    for old, new in TONE.items():
        s = s.replace(old, new)
    return s


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot((a[1] - b[1]) * 111.32, (a[0] - b[0]) * 111.32 * math.cos(math.radians((a[1] + b[1]) / 2)))


def write_boundaries(src: Path, out: Path, tolerance_deg: float = 0.0003) -> None:
    feats = []
    for folder, _ in QUOTA.values():
        prov = json.loads((src / folder / f"{folder}.geojson").read_text(encoding="utf-8"))["features"][0]["properties"]
        for f in sorted(glob.glob(str(src / folder / "wards" / "*.geojson"))):
            ft = json.loads(Path(f).read_text(encoding="utf-8"))["features"][0]
            pr, geom = ft["properties"], shape(ft["geometry"])
            c = geom.representative_point()
            simple = geom.simplify(tolerance_deg, preserve_topology=True)
            feats.append({"type": "Feature", "geometry": json.loads(json.dumps(simple.__geo_interface__)),
                          "properties": {"code": pr["code"], "name": pr["fullName"], "province": prov["fullName"],
                                         "province_code": prov["code"], "label": [round(c.x, 5), round(c.y, 5)],
                                         "area_km2": pr["areaKm2"]}})
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False)
    out.write_text(re.sub(r"(\d+\.\d{5})\d+", r"\1", text), encoding="utf-8")  # 5 decimals (~1 m)
    print(f"{len(feats)} communes -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geojson", type=Path, default=Path("geojson"))
    ap.add_argument("--out", type=Path, default=Path("data/communes_notable.csv"))
    ap.add_argument("--boundaries", type=Path, default=Path("data/boundaries/communes_2025.geojson"))
    args = ap.parse_args()
    write_boundaries(args.geojson, args.boundaries)

    communes: dict[str, dict] = {}  # code -> unit (within range)
    for prov, (folder, _) in QUOTA.items():
        for f in glob.glob(str(args.geojson / folder / "wards" / "*.geojson")):
            ft = json.loads(Path(f).read_text(encoding="utf-8"))["features"][0]
            pr, geom = ft["properties"], shape(ft["geometry"])
            c = geom.representative_point()
            r = km((c.x, c.y), RADAR)
            if r <= MAX_RANGE_KM:
                communes[pr["code"]] = {"province": prov, "code": pr["code"], "name": pr["fullName"], "short": pr["name"],
                                        "lon": c.x, "lat": c.y, "km": r, "geom": geom}
    by_name = {}
    for u in communes.values():
        by_name.setdefault((u["province"], norm(u["short"])), []).append(u)

    chosen: dict[str, tuple[dict, str, str]] = {}  # code -> (unit, kind, check)
    reader = shapefile.Reader(str(GADM_DISTRICTS))
    fields = [f[0] for f in reader.fields[1:]]
    former = set(CORES)
    for rec in reader.records():
        d = dict(zip(fields, rec))
        if d["ENGTYPE_2"] in ("City", "Town") and d["NAME_1"] in MERGED:
            former.add((MERGED[d["NAME_1"]], re.sub(r"\s*\((?:Thành phố|Thị xã)\)$", "", d["NAME_2"])))
    for prov, name in sorted(former):
        match = by_name.get((prov, norm(name)), [])
        match = [u for u in match if u["name"].startswith("Phường")] or match
        if match:
            chosen[match[0]["code"]] = (match[0], "centre", "")
        elif (prov, name) in RENAMED:
            lon, lat, unsure = RENAMED[(prov, name)]
            hit = next((u for u in communes.values() if u["province"] == prov and u["geom"].contains(Point(lon, lat))), None)
            if hit is not None:
                chosen[hit["code"]] = (hit, "centre", f"centre of former {name}? (renamed in 2025)" if unsure else "")
        # else: beyond range, or no ward of that name
    for prov, full in CITY:
        u = next((u for u in communes.values() if u["province"] == prov and norm(u["name"]) == norm(full)), None)
        if u is not None:
            chosen[u["code"]] = (u, "city", "")

    def scale(r: float) -> float:
        return max(3.0, 0.18 * r)

    for prov, (_, quota) in QUOTA.items():
        taken = [c for c, (u, _, _) in chosen.items() if u["province"] == prov]
        pool = [u for u in communes.values() if u["province"] == prov and u["code"] not in chosen and not u["name"].startswith("Đặc khu")]
        while len(taken) < quota and pool:
            best = max(pool, key=lambda u: min(km((u["lon"], u["lat"]), (communes[t]["lon"], communes[t]["lat"])) for t in taken)
                       / scale(u["km"]) if taken else -u["km"])
            chosen[best["code"]] = (best, "spread", "")
            taken.append(best["code"])
            pool.remove(best)

    order = list(QUOTA)
    rows = sorted(chosen.values(), key=lambda x: (order.index(x[0]["province"]), x[0]["km"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["province", "code", "name", "kind", "lon", "lat", "km_from_radar", "check"])
        for u, kind, check in rows:
            w.writerow([u["province"], u["code"], u["name"], kind, f"{u['lon']:.4f}", f"{u['lat']:.4f}", f"{u['km']:.0f}", check])
    counts = {p: sum(1 for u, _, _ in rows if u["province"] == p) for p in order}
    print(f"{len(rows)} communes -> {args.out}: {counts}")


if __name__ == "__main__":
    main()
