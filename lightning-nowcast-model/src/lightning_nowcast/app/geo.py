"""Map geometry: the forecast grid resampled to longitude / latitude, boundaries, the communes and range rings."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..grid import (
    CENTER_LAT,
    CENTER_LON,
    OUTER,
    RES_KM,
    aeqd_km,
    aeqd_lonlat,
    grid_lonlat,
)
from ..regions import Commune, all_provinces, communes, country, neighbours

LONLAT_RES = 0.018  # degrees per display cell (~2 km)
KM_PER_DEG = 111.32


@dataclass(frozen=True)
class Raster:
    """Nearest grid cell of every display cell of a regular longitude / latitude raster (rows south -> north)."""
    extent: tuple[float, float, float, float]  # lon0, lon1, lat0, lat1
    iy: np.ndarray
    ix: np.ndarray
    valid: np.ndarray

    @classmethod
    def build(cls) -> Raster:
        lon, lat = grid_lonlat()
        lons = np.arange(lon.min(), lon.max() + LONLAT_RES, LONLAT_RES)
        lats = np.arange(lat.min(), lat.max() + LONLAT_RES, LONLAT_RES)
        glon, glat = np.meshgrid(lons, lats)
        iy, ix = cell_of(glon, glat)
        valid = (iy >= 0) & (iy < OUTER) & (ix >= 0) & (ix < OUTER)
        half = LONLAT_RES / 2
        return cls((lons[0] - half, lons[-1] + half, lats[0] - half, lats[-1] + half), np.clip(iy, 0, OUTER - 1), np.clip(ix, 0, OUTER - 1), valid)

    def sample(self, field: np.ndarray) -> np.ndarray:
        """A (320, 320) field on the display raster; NaN outside the grid."""
        out = field[self.iy, self.ix].astype(np.float32)
        out[~self.valid] = np.nan
        return out


def cell_of(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the grid cell nearest to each point (may fall outside the grid)."""
    x, y = aeqd_km(lat, lon)
    half = (OUTER - 1) / 2
    return np.round(np.asarray(y) / RES_KM + half).astype(int), np.round(np.asarray(x) / RES_KM + half).astype(int)


def ring(radius_km: float, n: int = 240) -> np.ndarray:
    """A circle about the radar."""
    a = np.linspace(0, 2 * np.pi, n)
    lon, lat = aeqd_lonlat(radius_km * np.sin(a), radius_km * np.cos(a))
    return np.column_stack([lon, lat])


def circle(lon: float, lat: float, radius_km: float, n: int = 72) -> np.ndarray:
    """A circle about a place (local flat-earth approximation: < 0.1% error at 25 km)."""
    a = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([lon + radius_km * np.sin(a) / (KM_PER_DEG * math.cos(math.radians(lat))), lat + radius_km * np.cos(a) / KM_PER_DEG])


@dataclass
class Geo:
    raster: Raster
    communes: list[Commune]     # bulletin order
    outlines: list[np.ndarray]  # all provinces in view
    coast: list[np.ndarray]
    foreign: list[np.ndarray]   # the neighbouring countries' land
    rings: dict[int, np.ndarray]
    center: tuple[float, float]

    def __post_init__(self) -> None:
        self.by_code = {c.code: c for c in self.communes}
        self._boxes = np.array([[min(r[:, 0].min() for r in c.rings), max(r[:, 0].max() for r in c.rings),
                                 min(r[:, 1].min() for r in c.rings), max(r[:, 1].max() for r in c.rings)] for c in self.communes])

    @classmethod
    def load(cls, boundaries: Path, notable_csv: Path) -> Geo:
        return cls(Raster.build(), communes(boundaries, notable_csv),
                   [r for a in all_provinces(boundaries) for r in a.rings], list(country(boundaries)), list(neighbours(boundaries)),
                   {r: ring(r) for r in (120, 300)}, (CENTER_LON, CENTER_LAT))

    def commune_at(self, lon: float, lat: float) -> Commune | None:
        b = self._boxes
        for i in np.flatnonzero((b[:, 0] <= lon) & (lon <= b[:, 1]) & (b[:, 2] <= lat) & (lat <= b[:, 3])):
            if self.communes[i].contains(np.array([lon]), np.array([lat]))[0]:
                return self.communes[i]
        return None

    def bounds(self, c: Commune) -> tuple[float, float, float, float]:
        """lon0, lon1, lat0, lat1 of a commune."""
        return tuple(self._boxes[self.communes.index(c)])
