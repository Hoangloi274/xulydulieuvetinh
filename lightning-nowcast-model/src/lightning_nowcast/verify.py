"""Verification the way the warning service uses the product: virtual stations in the three regions.

    uv run python -m lightning_nowcast.verify --cache model=runs/cache_2026.npz \\
        --strokes-csv data/lightning/strokes_2025-01_2026-09_300km.csv --duty-hours 7-19 --out runs/verify/2026

Virtual stations are a regular grid (--spacing-km) of points inside each region (Ho Chi Minh City, South East, South
West: the provinces as merged in 2025, see regions.py) and within --max-range-km of the radar. Hand-picked places
give few, biased samples; a regular grid scores every place alike. At each station and issue time (anchor + 10 min)
the forecast is the probability at the station's pixel; the observation is a stroke within the window's radius of
the station during the window, taken straight from the stroke list. Reported, per model, region and window, over
a threshold sweep:

  table      hits / misses / false alarms / correct negatives over (station, issue time), with POD, FAR, CSI,
             frequency bias and percent correct (the scores of the project proposal); persistence (a stroke within
             radius in the 10 min before the issue time) is scored alongside.
  onset      first stroke within radius after >= --quiet-min without one ("the storm arrives"): the share warned by
             any issue time whose window contains it, and the share warned at least 10 / 20 / 30 min ahead.
             Persistence cannot warn an onset, so this is the part of the job only the model does.
  episodes   runs of consecutive issue times with a warning at a station; an episode is false if no stroke falls
             within radius from its first issue time to the end of its last window. False episodes per station-day.
  region     "lightning anywhere in the region": forecast = highest station probability, observed = any station.
  reliability  observed frequency per forecast-probability bin, and the Brier skill score against the base rate.

Several caches (forecast_cache outputs, NAME=PATH) are scored on the issue times they all have.
--duty-hours keeps issue times within the duty shift (local time), e.g. 7-19. Outputs: <out>.json, <out>.md.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .grid import RES_KM, aeqd_km, aeqd_lonlat
from .regions import DEFAULT_BOUNDARIES, REGIONS, province_masks, region_of

SWEEP = np.round(np.arange(0.05, 0.91, 0.05), 2)
LEADS_MIN = (10, 20, 30)
ICT = np.timedelta64(7, "h")
ISSUE_DELAY = np.timedelta64(10, "m")  # windows (T+10, T+W]: the forecast is issued at T+10


@dataclass(frozen=True)
class Stations:
    region: np.ndarray  # (S,) region name
    x: np.ndarray  # km east of the radar (azimuthal equidistant, as the grid)
    y: np.ndarray


@dataclass
class Cache:
    """A forecast_cache output."""
    anchors: np.ndarray  # (N,) datetime64[m]
    prob: np.ndarray  # (N, K, n, n)
    windows_min: list[int]
    radii_km: list[float]
    thresholds: list[float]

    @classmethod
    def load(cls, path: Path) -> Cache:
        d = np.load(path)
        return cls(d["anchors"].astype("datetime64[m]"), d["prob"], [int(w) for w in d["windows_min"]],
                   [float(r) for r in d["radii_km"]], [float(t) for t in d["thresholds"]])

    @property
    def names(self) -> list[str]:
        return [f"{w}r{r:g}" for w, r in zip(self.windows_min, self.radii_km)]

    def at_stations(self, st: Stations, anchors: np.ndarray) -> np.ndarray:
        """(S, len(anchors), K) probability at each station's pixel; zero where the cache has no forecast."""
        n = self.prob.shape[-1]
        half = (n - 1) / 2.0
        ix = np.clip(np.round(st.x / RES_KM + half), 0, n - 1).astype(int)
        iy = np.clip(np.round(st.y / RES_KM + half), 0, n - 1).astype(int)
        out = np.zeros((st.x.size, anchors.size, len(self.windows_min)), np.float32)
        out[:, np.searchsorted(anchors, self.anchors)] = np.moveaxis(self.prob[:, :, iy, ix].astype(np.float32), 2, 0)
        return out


def make_stations(boundaries: Path, spacing_km: float, max_range_km: float) -> Stations:
    g = np.arange(-max_range_km, max_range_km + 1e-6, spacing_km)
    xx, yy = np.meshgrid(g, g)
    keep = np.hypot(xx, yy) <= max_range_km
    xx, yy = xx[keep], yy[keep]
    region = np.full(xx.size, "", dtype=object)
    for r, inside in region_of(province_masks(boundaries, *aeqd_lonlat(xx, yy))).items():
        region[inside & (region == "")] = r
    sel = region != ""
    return Stations(region[sel].astype(str), xx[sel], yy[sel])


def station_strokes(csv: Path, st: Stations, radius_km: float, t0: np.datetime64, t1: np.datetime64) -> list[np.ndarray]:
    """Sorted stroke times (datetime64[s]) within radius of each station."""
    df = pd.read_csv(csv, usecols=["time_utc", "lat", "lon"])
    t = pd.to_datetime(df["time_utc"], utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[s]")
    keep = (t >= t0) & (t <= t1)
    t, lat, lon = t[keep], df["lat"].to_numpy()[keep], df["lon"].to_numpy()[keep]
    order = np.argsort(t)
    t, (x, y) = t[order], aeqd_km(lat[order], lon[order])
    out = []
    for sx, sy in zip(st.x, st.y):
        near = (np.abs(x - sx) <= radius_km) & (np.abs(y - sy) <= radius_km)
        near[near] = np.hypot(x[near] - sx, y[near] - sy) <= radius_km
        out.append(np.unique(t[near]))
    return out


def any_between(times: list[np.ndarray], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(S, N) bool: a stroke at station s in (lo[n], hi[n]]."""
    return np.stack([np.searchsorted(ts, hi, "right") > np.searchsorted(ts, lo, "right") for ts in times])


def scores(h: float, m: float, f: float, cn: float) -> dict[str, Any]:
    div = lambda a, b: float(a / b) if b else None
    return {"hits": int(h), "misses": int(m), "false_alarms": int(f), "correct_negatives": int(cn),
            "POD": div(h, h + m), "FAR": div(f, h + f), "CSI": div(h, h + m + f), "bias": div(h + f, h + m),
            "PC": div(h + cn, h + m + f + cn)}


def table(fc: np.ndarray, ob: np.ndarray) -> dict[str, Any]:
    return scores((fc & ob).sum(), (~fc & ob).sum(), (fc & ~ob).sum(), (~fc & ~ob).sum())


def evaluate(prob_s: np.ndarray, ok: np.ndarray, times: dict[float, list[np.ndarray]], radii: list[float], issue: np.ndarray,
             ends: list[np.ndarray], mask_s: np.ndarray, duty: np.ndarray, quiet_min: int, n_days: float) -> dict[str, Any]:
    """prob_s: (S, N, K) station probabilities; ok: (N,) anchors with a forecast; mask_s: (S,) stations of the region;
    times[r]: stroke times within r km of each station; radii[k]: the radius of window k."""
    use_t = ok & duty
    sel_s = np.flatnonzero(mask_s)
    res: dict[str, Any] = {"stations": int(sel_s.size), "issue_times": int(use_t.sum()), "windows": []}
    for k in range(prob_s.shape[2]):
        tm = [times[radii[k]][s] for s in sel_s]
        pk = prob_s[sel_s][:, use_t, k]
        obs = any_between(tm, issue[use_t], ends[k][use_t])
        pers = any_between(tm, issue[use_t] - np.timedelta64(10, "m"), issue[use_t])
        base = float(obs.mean())
        bs_ref = base * (1 - base)
        edges = np.linspace(0, 1, 11)
        b = np.clip(np.digitize(pk, edges) - 1, 0, 9)
        rel = [{"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int((b == i).sum()),
                "forecast": float(pk[b == i].mean()) if (b == i).any() else None,
                "observed": float(obs[b == i].mean()) if (b == i).any() else None} for i in range(10)]
        w = {"base_rate": base, "BSS": 1 - float(((pk - obs) ** 2).mean()) / bs_ref if bs_ref else None, "reliability": rel,
             "persistence": table(pers, obs), "sweep": []}
        reg_obs, reg_p = obs.any(axis=0), pk.max(axis=0)
        for thr in SWEEP:
            fc = pk >= thr
            row = {"threshold": float(thr), **table(fc, obs), "region_any": table(reg_p >= thr, reg_obs)}
            row.update(onsets(fc, use_t, tm, issue, ends[k], quiet_min, ok))
            row.update(episodes(fc, tm, issue[use_t], ends[k][use_t], n_stations=sel_s.size, n_days=n_days))
            w["sweep"].append(row)
        res["windows"].append(w)
    return res


def onsets(warn: np.ndarray, use_t: np.ndarray, times: list[np.ndarray], issue: np.ndarray, end: np.ndarray,
           quiet_min: int, ok: np.ndarray) -> dict[str, Any]:
    """Share of onsets warned by an issue time whose window contains them, and with lead >= LEADS_MIN."""
    idx = np.flatnonzero(use_t)
    n_ev, warned, lead_hits = 0, 0, np.zeros(len(LEADS_MIN))
    leads = []
    lo, hi = issue[idx[0]], end[idx[-1]]
    for s, ts in enumerate(times):
        if ts.size == 0:
            continue
        gaps = np.diff(ts, prepend=ts[0] - np.timedelta64(quiet_min * 60 + 1, "s"))
        on = ts[gaps > np.timedelta64(quiet_min * 60, "s")]
        on = on[(on > lo) & (on <= hi)]
        for tau in on:
            cand = (issue[idx] < tau) & (end[idx] >= tau)  # issue times whose window contains the onset
            if not cand.any() or not ok[idx][cand].all():
                continue  # a forecast is missing for this onset: not scored
            n_ev += 1
            w = cand & warn[s]
            if w.any():
                warned += 1
                lead = (tau - issue[idx][w].min()) / np.timedelta64(1, "m")
                leads.append(lead)
                lead_hits += lead >= np.array(LEADS_MIN)
    return {"onsets": n_ev, "onset_POD": warned / n_ev if n_ev else None,
            **{f"onset_POD_lead{L}": float(h / n_ev) if n_ev else None for L, h in zip(LEADS_MIN, lead_hits)},
            "onset_median_lead_min": float(np.median(leads)) if leads else None}


def episodes(fc: np.ndarray, times: list[np.ndarray], issue: np.ndarray, end: np.ndarray, n_stations: int, n_days: float) -> dict[str, Any]:
    """Warning episodes = runs of consecutive (10-min apart) issue times with a warning, per station."""
    n_ep = n_false = 0
    step = np.timedelta64(10, "m")
    for s, ts in enumerate(times):
        on = np.flatnonzero(fc[s])
        if on.size == 0:
            continue
        brk = np.flatnonzero(np.diff(issue[on]) > step) + 1
        for run in np.split(on, brk):
            n_ep += 1
            a, b = issue[run[0]], end[run[-1]]
            if np.searchsorted(ts, b, "right") <= np.searchsorted(ts, a, "right"):
                n_false += 1
    return {"episodes": n_ep, "episode_FAR": n_false / n_ep if n_ep else None,
            "false_episodes_per_station_day": n_false / (n_stations * n_days) if n_days else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", nargs="+", required=True, help="NAME=forecast_cache.npz (each scored on its own windows)")
    ap.add_argument("--strokes-csv", type=Path, required=True)
    ap.add_argument("--boundaries-dir", type=Path, default=DEFAULT_BOUNDARIES)
    ap.add_argument("--spacing-km", type=float, default=10.0)
    ap.add_argument("--max-range-km", type=float, default=250.0, help="Stations must lie inside the model domain (~260 km)")
    ap.add_argument("--quiet-min", type=int, default=30, help="Onset = first stroke after this many minutes without one")
    ap.add_argument("--duty-hours", type=str, default=None, help="Local (ICT) issue hours kept, e.g. 7-19")
    ap.add_argument("--out", type=Path, required=True, help="Output prefix (.json and .md are added)")
    args = ap.parse_args()

    caches = {name: Cache.load(Path(path)) for name, path in (spec.split("=", 1) for spec in args.cache)}
    # common 10-min issue grid over the union of anchors; a model scores only where every model has a forecast
    all_a = np.unique(np.concatenate([c.anchors for c in caches.values()]))
    anchors = np.arange(all_a[0], all_a[-1] + np.timedelta64(10, "m"), np.timedelta64(10, "m"))
    common = np.ones(anchors.size, bool)
    for c in caches.values():
        common &= np.isin(anchors, c.anchors)
    issue = anchors + ISSUE_DELAY
    if args.duty_hours:
        hours = (issue + ICT).astype("datetime64[h]").astype(int) % 24
        h0, h1 = (int(v) for v in args.duty_hours.split("-"))
        duty = (hours >= h0) & (hours < h1)
    else:
        duty = np.ones(anchors.size, bool)
    n_days = float((common & duty).sum()) * 10 / (60 * 24)
    longest = max(w for c in caches.values() for w in c.windows_min)

    st = make_stations(args.boundaries_dir, args.spacing_km, args.max_range_km)
    times = {r: station_strokes(args.strokes_csv, st, r, anchors[0] - np.timedelta64(2, "h"), anchors[-1] + np.timedelta64(longest, "m"))
             for r in sorted({r for c in caches.values() for r in c.radii_km})}
    out: dict[str, Any] = {"quiet_min": args.quiet_min, "duty_hours": args.duty_hours, "spacing_km": args.spacing_km,
                           "max_range_km": args.max_range_km, "first_anchor": str(anchors[0]), "last_anchor": str(anchors[-1]),
                           "issue_times_all_models": int((common & duty).sum()), "models": {}, "thresholds_in_ckpt": {}, "window_names": {}}
    for name, c in caches.items():
        prob_s = c.at_stations(st, anchors)
        ends = [anchors + np.timedelta64(w, "m") for w in c.windows_min]
        out["thresholds_in_ckpt"][name] = c.thresholds
        out["window_names"][name] = c.names
        out["models"][name] = {r: evaluate(prob_s, common, times, c.radii_km, issue, ends, st.region == r, duty, args.quiet_min, n_days)
                               for r in REGIONS if (st.region == r).any()}
        print(f"{name}: scored", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=1))
    args.out.with_suffix(".md").write_text(report(out))
    print(report(out))


def _f(v: float | None, fmt: str = ".2f") -> str:
    return "–" if v is None else format(v, fmt)


def report(out: dict[str, Any]) -> str:
    lines = [f"# Virtual-station verification {out['first_anchor']} – {out['last_anchor']} UTC",
             (f"Stations every {out['spacing_km']:g} km within {out['max_range_km']:g} km of the radar; onset after "
              f"{out['quiet_min']} quiet min; duty hours {out['duty_hours'] or 'all'}; {out['issue_times_all_models']} issue times."), ""]
    for name, regions in out["models"].items():
        thr_ck = out["thresholds_in_ckpt"][name]
        for r, res in regions.items():
            lines.append(f"## {name} — {r} ({res['stations']} stations)")
            for k, w in enumerate(res["windows"]):
                W, R = out["window_names"][name][k].split("r")
                lines.append(f"\n**Window (T+10, T+{W}] within {R} km** base rate {w['base_rate']:.4f}, "
                             f"BSS {_f(w['BSS'], '+.3f')}; persistence POD {_f(w['persistence']['POD'])} FAR {_f(w['persistence']['FAR'])} "
                             f"CSI {_f(w['persistence']['CSI'])}; checkpoint threshold {thr_ck[k]:.2f}\n")
                lines.append("| thr | POD | FAR | CSI | bias | onsets | onset POD | ≥10 min | ≥20 min | ≥30 min | median lead | "
                             "false episodes /stn/day | region POD | region FAR |")
                lines.append("|" + "---|" * 14)
                for row in w["sweep"]:
                    ra = row["region_any"]
                    lines.append(f"| {row['threshold']:.2f} | {_f(row['POD'])} | {_f(row['FAR'])} | {_f(row['CSI'])} | {_f(row['bias'])} | "
                                 f"{row['onsets']} | {_f(row['onset_POD'])} | {_f(row['onset_POD_lead10'])} | {_f(row['onset_POD_lead20'])} | "
                                 f"{_f(row['onset_POD_lead30'])} | {_f(row['onset_median_lead_min'], '.0f')} | "
                                 f"{_f(row['false_episodes_per_station_day'], '.3f')} | {_f(ra['POD'])} | {_f(ra['FAR'])} |")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
