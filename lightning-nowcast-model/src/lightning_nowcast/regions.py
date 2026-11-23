"""The wards and communes (phường, xã, đặc khu: the units after the 1 July 2025 reform) the forecast is issued for, as
the Southern bulletins list them, and the outlines for the map.

    data/boundaries/communes_2025.geojson    the communes of the nine provinces (scripts/select_communes.py)
    data/communes_notable.csv                the communes labelled and listed first (scripts/select_communes.py)
    data/boundaries/provinces_2025.geojson   province outlines, and the three regions of verify.py (scripts/build_boundaries.py)
    data/boundaries/vietnam_2025.geojson     the national outline (scripts/build_boundaries.py)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

REGIONS = {
    "HCMC": ["TP. Hồ Chí Minh"],
    "South East": ["Đồng Nai", "Tây Ninh"],
    "South West": ["Đồng Tháp", "Vĩnh Long", "An Giang", "Cần Thơ", "Cà Mau"],
}
REGION_OF = {p: r for r, ps in REGIONS.items() for p in ps}
DATA = Path(__file__).resolve().parents[2] / "data"
DEFAULT_BOUNDARIES = DATA / "boundaries"
DEFAULT_NOTABLE = DATA / "communes_notable.csv"
# the provinces of the Southern bulletins, in their order (west to east), by province code
PROVINCE_ORDER = ("96", "92", "91", "86", "82", "80", "79", "75", "68")


@dataclass(frozen=True)
class Area:
    """A province: its outline as rings of (lon, lat) points, inside by the even-odd rule."""
    name: str
    rings: tuple[np.ndarray, ...]

    def contains(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        pts = np.column_stack([np.ravel(lon), np.ravel(lat)])
        inside = np.zeros(len(pts), bool)
        for ring in self.rings:
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            near = np.flatnonzero((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
            if near.size:
                inside[near] ^= MplPath(ring).contains_points(pts[near])
        return inside.reshape(np.shape(lon))


@dataclass(frozen=True)
class Commune(Area):
    """A ward or commune: its warning is the highest probability over its grid cells (lightning within the window's
    radius of some point of it)."""
    code: str = ""
    province: str = ""       # full name, as the bulletins write it ("Tỉnh Đồng Tháp")
    province_code: str = ""
    lon: float = 0.0         # label point (inside the commune)
    lat: float = 0.0
    notable: bool = False    # labelled and listed first


def _rings(geometry: dict) -> tuple[np.ndarray, ...]:
    polys = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    return tuple(np.asarray(ring, float) for poly in polys for ring in poly)


def _features(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def all_provinces(boundaries: Path) -> list[Area]:
    """Every province in the map's extent (outlines for display)."""
    return [Area(f["properties"]["name"], _rings(f["geometry"])) for f in _features(boundaries / "provinces_2025.geojson")]


def provinces(boundaries: Path) -> list[Area]:
    """The provinces of REGIONS."""
    out = [a for a in all_provinces(boundaries) if a.name in REGION_OF]
    if missing := set(REGION_OF) - {a.name for a in out}:
        raise ValueError(f"provinces not found in {boundaries}: {sorted(missing)}")
    return out


def country(boundaries: Path) -> tuple[np.ndarray, ...]:
    """Rings of the national outline (coast and borders) within the map's extent."""
    return tuple(r for f in _features(boundaries / "vietnam_2025.geojson") for r in _rings(f["geometry"]))


def neighbours(boundaries: Path) -> tuple[np.ndarray, ...]:
    """Rings of the neighbouring countries' land (Natural Earth 1:10m, clipped to the region: land_ne10m.geojson)."""
    return tuple(r for f in _features(boundaries / "land_ne10m.geojson") if f["properties"]["iso"] != "VNM" for r in _rings(f["geometry"]))


def communes(boundaries: Path = DEFAULT_BOUNDARIES, notable_csv: Path = DEFAULT_NOTABLE) -> list[Commune]:
    """Every commune of the nine provinces, in bulletin order (province west to east, then name)."""
    with notable_csv.open(encoding="utf-8") as f:
        notable = {r["code"] for r in csv.DictReader(f)}
    out = []
    for f in _features(boundaries / "communes_2025.geojson"):
        p = f["properties"]
        out.append(Commune(p["name"], _rings(f["geometry"]), p["code"], p["province"], p["province_code"],
                           float(p["label"][0]), float(p["label"][1]), p["code"] in notable))
    rank = {c: i for i, c in enumerate(PROVINCE_ORDER)}
    return sorted(out, key=lambda c: (rank.get(c.province_code, len(rank)), c.name))


def commune_masks(units: list[Commune], lon: np.ndarray, lat: np.ndarray, inside: np.ndarray) -> dict[str, np.ndarray]:
    """Commune code -> its cells of the grid (lon, lat), among `inside`: the cells whose centre lies in it, or, for a ward
    too small to hold a cell centre, the cell nearest its label point. Communes with no cell inside are left out."""
    out = {}
    for c in units:
        m = c.contains(lon, lat) & inside
        if not m.any():
            iy, ix = np.unravel_index(np.argmin((lon - c.lon) ** 2 + ((lat - c.lat) * 1.0) ** 2), lon.shape)
            if not inside[iy, ix] or np.hypot((lon[iy, ix] - c.lon) * 109, (lat[iy, ix] - c.lat) * 111) > 3:
                continue
            m = np.zeros(lon.shape, bool)
            m[iy, ix] = True
        out[c.code] = m
    return out


def province_masks(boundaries: Path, lon: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """Province name (as in REGIONS) -> which of the points (lon, lat, any equal shapes) lie inside it."""
    return {a.name: a.contains(lon, lat) for a in provinces(boundaries)}


def region_of(provinces: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Region name -> union of its provinces' masks."""
    return {r: np.any([provinces[p] for p in ps], axis=0) for r, ps in REGIONS.items()}
