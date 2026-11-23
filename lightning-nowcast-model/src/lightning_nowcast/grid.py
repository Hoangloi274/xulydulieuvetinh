"""The forecast grid and the input sources.

Grids are azimuthal equidistant about the Nhà Bè radar with 2 km cells, rows south -> north:
  INNER  256 x 256 (512 km): the 120 km radar volume and the satellite
  OUTER  320 x 320 (640 km): the 300 km radar volume, lightning, targets and forecasts
The two share their cell centres; INNER sits at offset OFFSET inside OUTER.
"""
from __future__ import annotations

import math
from enum import StrEnum

import numpy as np

CENTER_LAT, CENTER_LON = 10.6595, 106.7283
RES_KM = 2.0
INNER, OUTER = 256, 320
OFFSET = (OUTER - INNER) // 2
EARTH_KM = 6371.0088


class Source(StrEnum):
    """Model inputs. String-valued, so they serialize as plain names in checkpoints and stores."""
    RADAR120 = "r120"   # 120 km volume, 8 elevations: column fields
    RADAR300 = "r300"   # 300 km long-range volume, 4 elevations
    SATELLITE = "sat"   # Himawari brightness temperatures
    LIGHTNING = "ltg"   # in-cloud and ground stroke density


R120_VARS = ("Z", "CMAX", "Z_M10", "Z_M20", "ET30", "VIL")
R300_VARS = ("Z", "CMAX", "Z_M10", "ET30", "VIL")
SAT_BANDS = ("B08", "B10", "B13", "B15", "B16")


def aeqd_km(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spherical azimuthal equidistant x (east), y (north) in km about the radar."""
    p0, l0 = np.deg2rad(CENTER_LAT), np.deg2rad(CENTER_LON)
    p, dl = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float)) - l0
    cosc = np.clip(np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(dl), -1, 1)
    c = np.arccos(cosc)
    k = np.where(c > 1e-9, c / np.maximum(np.sin(c), 1e-12), 1.0)
    return EARTH_KM * k * np.cos(p) * np.sin(dl), EARTH_KM * k * (np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(dl))


def aeqd_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of aeqd_km: longitude and latitude of points x (east), y (north) km from the radar."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = np.hypot(x, y)
    cc = rho / EARTH_KM
    p0, l0 = math.radians(CENTER_LAT), math.radians(CENTER_LON)
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.cos(cc) * math.sin(p0) + np.where(rho > 0, y * np.sin(cc) * math.cos(p0) / rho, 0.0))
        lon = l0 + np.arctan2(x * np.sin(cc), rho * math.cos(p0) * np.cos(cc) - y * math.sin(p0) * np.sin(cc))
    return np.degrees(lon), np.degrees(lat)


def grid_lonlat(n: int = OUTER) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of the cell centres of the n x n grid."""
    c = (np.arange(n) - (n - 1) / 2.0) * RES_KM
    return aeqd_lonlat(*np.meshgrid(c, c))


def range_km_map(n: int = OUTER) -> np.ndarray:
    """Distance of each cell centre from the radar, km."""
    c = (n - 1) / 2.0
    y, x = np.mgrid[:n, :n]
    return (np.hypot(x - c, y - c) * RES_KM).astype(np.float32)
