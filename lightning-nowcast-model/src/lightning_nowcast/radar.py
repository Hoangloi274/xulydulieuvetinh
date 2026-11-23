"""Column (3-D) radar features from Nha Be SIGMET/IRIS volumes, gridded to the 2 km nowcast grid.

The 120 km volume holds 8 elevations (0.5-12 deg), but the original pipeline kept only
the 0.5 deg sweep. Lightning needs charge separation in the mixed-phase region, so the
classic radar predictors look aloft: reflectivity >= 35-40 dBZ at the -10 C level
(Gremillion & Orville 1999; Mosier et al. 2011), echo-top heights and
vertically integrated liquid/ice (Greene & Clark 1972; Mosier et al. 2011).

Per volume this module returns, on the GridSpec grid:
  Z        lowest-sweep reflectivity (identical to the original `Z`), dBZ
  Q        1 where the lowest sweep has a gate in the cell
  CMAX     column maximum reflectivity over all sweeps, dBZ (0 where no echo)
  Z_M10    reflectivity interpolated to the -10 C height (default 6.5 km), NaN if unsampled
  Z_M20    reflectivity interpolated to the -20 C height (default 8.0 km), NaN if unsampled
  ET30     highest beam-centre height with >= 30 dBZ, km (0 if none)
  VIL      vertically integrated liquid, kg m-2 (Greene & Clark 1972, dBZ capped at 56)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EFFECTIVE_EARTH_RADIUS_M = 4.0 / 3.0 * 6371000.0
FIELDS_3D = ("CMAX", "Z_M10", "Z_M20", "ET30", "VIL")
FIELD_CANDIDATES = ("DBZH", "DB_DBZ", "DB_DBZ2", "DBTH")


@dataclass(frozen=True)
class GridSpec:
    center_lat: float = 10.6595
    center_lon: float = 106.7283
    nx: int = 256
    ny: int = 256
    res_m: float = 2000.0
    max_range_km: float = 120.0

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.nx, dtype=np.float32) - self.nx / 2 + 0.5) * self.res_m

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.ny, dtype=np.float32) - self.ny / 2 + 0.5) * self.res_m

    def ground_range_m(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.x, self.y)
        return np.hypot(xx, yy)


@dataclass
class Sweep:
    elevation_deg: float
    z: np.ndarray  # (azimuth, range) dBZ
    azimuth_deg: np.ndarray
    range_m: np.ndarray
    elevation_per_ray: np.ndarray


def read_volume(path: Path, target_range_km: float = 120.0, tol_km: float = 30.0) -> list[Sweep] | None:
    """All sweeps of a volume, or None if it is not the `target_range_km` task."""
    import xradar as xd  # heavy import; only needed when converting

    tree = xd.io.open_iris_datatree(str(path))
    sweeps: list[Sweep] = []
    for name in sorted((k for k in tree.children if k.startswith("sweep_")), key=lambda k: int(k.split("_")[1])):
        ds = tree[name].ds
        rg = np.asarray(ds["range"].values, dtype=np.float32)
        if abs(rg.max() / 1000.0 - target_range_km) > tol_km:
            return None
        field = next((f for f in FIELD_CANDIDATES if f in ds.variables), None)
        if field is None:
            raise KeyError(f"{path.name}: no reflectivity field in {list(ds.data_vars)}")
        az = np.asarray(ds["azimuth"].values, dtype=np.float32)
        el = np.asarray(ds["elevation"].values, dtype=np.float32)
        z = np.asarray(ds[field].transpose("azimuth", "range").values, dtype=np.float32)
        fixed = float(ds["sweep_fixed_angle"].values) if "sweep_fixed_angle" in ds else float(np.median(el))
        sweeps.append(Sweep(fixed, z, az, rg, np.broadcast_to(el, az.shape).astype(np.float32)))
    return sweeps


def grid_sweep(sw: Sweep, grid: GridSpec) -> np.ndarray:
    """Nearest-cell maximum of one sweep (same rule as the original converter); NaN where no gate."""
    keep = sw.range_m <= grid.max_range_km * 1000.0
    z = sw.z[:, keep]
    rg = sw.range_m[keep]
    az = np.deg2rad(sw.azimuth_deg)[:, None]
    ground = rg[None, :] * np.cos(np.deg2rad(sw.elevation_per_ray))[:, None]
    x = (ground * np.sin(az)).ravel()
    y = (ground * np.cos(az)).ravel()
    zv = z.ravel()
    ok = np.isfinite(zv)
    xi = np.floor((x[ok] - (grid.x[0] - grid.res_m / 2)) / grid.res_m).astype(np.int64)
    yi = np.floor((y[ok] - (grid.y[0] - grid.res_m / 2)) / grid.res_m).astype(np.int64)
    inb = (xi >= 0) & (xi < grid.nx) & (yi >= 0) & (yi < grid.ny)
    flat = np.full(grid.nx * grid.ny, -np.inf, dtype=np.float32)
    np.maximum.at(flat, yi[inb] * grid.nx + xi[inb], zv[ok][inb])
    out = flat.reshape(grid.ny, grid.nx)
    out[~np.isfinite(out)] = np.nan
    return out


def beam_height_m(ground_m: np.ndarray, elevation_deg: float) -> np.ndarray:
    """Beam-centre height above the radar for the 4/3 effective-earth model (Doviak & Zrnic 1993)."""
    el = np.deg2rad(elevation_deg)
    slant = ground_m / np.cos(el)
    a = EFFECTIVE_EARTH_RADIUS_M
    return np.sqrt(slant**2 + a**2 + 2.0 * slant * a * np.sin(el)) - a


def column_features(zk: np.ndarray, elevations: list[float], grid: GridSpec,
                    h_m10_m: float = 6500.0, h_m20_m: float = 8000.0) -> dict[str, np.ndarray]:
    """zk: (K, ny, nx) gridded sweeps (NaN = not sampled, very negative = no echo), elevations ascending."""
    ground = grid.ground_range_m()
    hk = np.stack([beam_height_m(ground, e) for e in elevations])  # (K, ny, nx)
    sampled = np.isfinite(zk)
    dbz = np.where(sampled, np.clip(zk, 0.0, 75.0), np.nan)  # no echo -> 0 dBZ

    def at_height(h0: float) -> np.ndarray:
        out = np.full(ground.shape, np.nan, dtype=np.float32)
        for k in range(len(elevations) - 1):
            lo, hi = hk[k], hk[k + 1]
            sel = (lo <= h0) & (h0 < hi) & sampled[k] & sampled[k + 1]
            w = (h0 - lo[sel]) / (hi[sel] - lo[sel])
            out[sel] = (1 - w) * dbz[k][sel] + w * dbz[k + 1][sel]
        return out

    cmax = np.nanmax(np.where(sampled, dbz, -np.inf), axis=0)
    cmax = np.where(np.isfinite(cmax), cmax, np.nan)
    et30 = np.max(np.where(sampled & (dbz >= 30.0), hk, 0.0), axis=0) / 1000.0
    # VIL = sum 3.44e-6 * mean(Z_lin)^(4/7) * dh, Z_lin in mm6 m-3, dBZ capped at 56 (Greene & Clark 1972)
    zlin = 10.0 ** (np.minimum(np.nan_to_num(dbz, nan=0.0), 56.0) / 10.0) * (dbz > 0)
    vil = np.zeros(ground.shape, dtype=np.float64)
    for k in range(len(elevations) - 1):
        both = sampled[k] & sampled[k + 1]
        layer = 3.44e-6 * ((zlin[k] + zlin[k + 1]) / 2.0) ** (4.0 / 7.0) * (hk[k + 1] - hk[k])
        vil += np.where(both, layer, 0.0)
    any_sampled = sampled.any(axis=0)
    return {
        "CMAX": cmax.astype(np.float32),
        "Z_M10": at_height(h_m10_m),
        "Z_M20": at_height(h_m20_m),
        "ET30": np.where(any_sampled, et30, np.nan).astype(np.float32),
        "VIL": np.where(any_sampled, vil, np.nan).astype(np.float32),
    }


class IncompleteVolume(ValueError):
    """A volume with fewer sweeps than expected: possibly still being written."""


def volume_features(path: Path, grid: GridSpec, min_sweeps: int = 0, **kw: float) -> dict[str, np.ndarray] | None:
    """Z, Q (lowest sweep, as before) plus the column features; None if `path` is not a volume of the grid's range.
    Raises IncompleteVolume if it has fewer than `min_sweeps` sweeps."""
    sweeps = read_volume(path, target_range_km=grid.max_range_km)
    if not sweeps:
        return None
    if len(sweeps) < min_sweeps:
        raise IncompleteVolume(f"{path.name}: {len(sweeps)} of {min_sweeps} sweeps")
    sweeps.sort(key=lambda s: s.elevation_deg)
    zk = np.stack([grid_sweep(s, grid) for s in sweeps])
    out = column_features(zk, [s.elevation_deg for s in sweeps], grid, **kw)
    out["Z"] = zk[0]
    out["Q"] = np.isfinite(zk[0]).astype(np.float32)
    return out
