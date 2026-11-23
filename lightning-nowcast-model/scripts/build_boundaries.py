# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyshp", "shapely>=2"]
# ///
"""Build the map boundaries from their sources (run again only to change them).

    uv run scripts/build_boundaries.py --hdx-zip vnm_admin_boundaries.shp.zip

Writes data/boundaries/provinces_2025.geojson, the provinces after the 1 July 2025 merger (34 in all; those in the
forecast map's extent are kept), and data/boundaries/vietnam_2025.geojson, the national outline. Both come from
OCHA/HDX "Viet Nam - Subnational Administrative Boundaries" (GSO, valid from 2025-09-25, CC BY-IGO):
https://data.humdata.org/dataset/cod-ab-vnm, simplified to ~200 m.

The communes and the notable list come from scripts/select_communes.py.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import shapefile
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
EXTENT = (103.5, 110.0, 7.4, 13.9)  # lon0, lon1, lat0, lat1: the forecast map and a margin
SIMPLIFY_DEG = 0.002
REGIONS = {
    "HCMC": ["TP. Hồ Chí Minh"],
    "South East": ["Đồng Nai", "Tây Ninh"],
    "South West": ["Đồng Tháp", "Vĩnh Long", "An Giang", "Cần Thơ", "Cà Mau"],
}


def read_shapes(path: Path):
    reader = shapefile.Reader(str(path))
    fields = [f[0] for f in reader.fields[1:]]
    for sr in reader.iterShapeRecords():
        if sr.shape is not None and sr.record is not None:
            yield dict(zip(fields, sr.record)), shape(sr.shape.__geo_interface__)


def write_geojson(path: Path, feats: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path} ({len(feats)} features, {path.stat().st_size // 1024} KB)")


def rounded(geom) -> dict:
    g = mapping(geom)
    def r(c):
        return [round(c[0], 4), round(c[1], 4)] if isinstance(c[0], float) else [r(x) for x in c]
    return {"type": g["type"], "coordinates": r(g["coordinates"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hdx-zip", type=Path, required=True, help="vnm_admin_boundaries.shp.zip from the HDX dataset")
    args = ap.parse_args()
    lon0, lon1, lat0, lat1 = EXTENT
    extent = box(lon0, lat0, lon1, lat1)

    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(args.hdx_zip).extractall(tmp)
        provinces = []
        for rec, geom in read_shapes(Path(tmp) / "vnm_admin1.shp"):
            if geom.intersects(extent):
                provinces.append((rec["adm1_name1"], rec["adm1_pcode"], geom))
        country = unary_union([g for _, g in read_shapes(Path(tmp) / "vnm_admin0.shp")])
    region_of = {p: r for r, ps in REGIONS.items() for p in ps}
    missing = set(region_of) - {n for n, _, _ in provinces}
    if missing:
        raise SystemExit(f"provinces not in the HDX data: {sorted(missing)}")
    write_geojson(ROOT / "data" / "boundaries" / "provinces_2025.geojson", [
        {"type": "Feature", "properties": {"name": n, "pcode": c, "region": region_of.get(n)},
         "geometry": rounded(g.simplify(SIMPLIFY_DEG, preserve_topology=True))} for n, c, g in provinces])
    write_geojson(ROOT / "data" / "boundaries" / "vietnam_2025.geojson", [
        {"type": "Feature", "properties": {"name": "Việt Nam"},
         "geometry": rounded(country.intersection(extent).simplify(SIMPLIFY_DEG, preserve_topology=True))}])


if __name__ == "__main__":
    main()
