"""One comparison picture per flagged forecast: our forecast at T, the hymetnet warnings available at T, and the strokes
that then fell, on one map, scored the same way.

    uv run python scripts/moment_panel.py --work-dir live 2026-10-23T04:20 [...]   (anchors T, UTC; default: all flagged)

Writes <work-dir>/moments/YYYYmmdd/HHMM_panel.png. Columns (times in giờ VN):
1. our 60-min forecast issued at T+9: probability of a stroke within 10 km during (T+10, T+60];
2. hymetnet's warnings as they were at our issue time (the latest issued before T+10; their leads +10..+60 min), the
   warned points valid in our window;
3. the strokes of the window: iWeather (the ✕ of the app) and, when its frame is saved, hymetnet's own.
Both columns are scored on the window both cover, (T+10, min(T+60, hymetnet's last lead)]: the share of iWeather
strokes in the warned area, and the warned area. Ours: cells >= 30% (the alert level). hymetnet's warnings are points
without an extent: the area within 10 km of a point, the radius our forecast uses.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from capture_moments import VN, fetch_time, page_frames, verify_page  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

from lightning_nowcast.app.geo import Geo, cell_of  # noqa: E402
from lightning_nowcast.app.style import (  # noqa: E402
    INK,
    INK_2,
    LEVEL_COLOR,
    MAP,
    MUTED,
    PROB_EDGES,
    RULE,
    short_name,  # noqa: E402
)
from lightning_nowcast.grid import OUTER, RES_KM  # noqa: E402
from lightning_nowcast.lightning_feed import CENTER_LAT as FEED_LAT  # noqa: E402
from lightning_nowcast.lightning_feed import CENTER_LON as FEED_LON
from lightning_nowcast.lightning_feed import RADIUS_KM as FEED_RADIUS_KM
from lightning_nowcast.lightning_feed import haversine_km, load_strokes  # noqa: E402
from lightning_nowcast.product import Forecast, Level  # noqa: E402
from lightning_nowcast.regions import DEFAULT_BOUNDARIES, DEFAULT_NOTABLE  # noqa: E402

ALERT = 0.3
RADIUS_KM = 10.0
HM_COLOR = "#c2410c"  # hymetnet's warned points (its own warning icon is orange-red)


def vn(t: datetime) -> str:
    return (t + VN).strftime("%H:%M")


def hymetnet_warnings(root: Path, anchor: datetime) -> tuple[datetime, dict[datetime, list[tuple[str, float, float]]]] | None:
    """The latest warnings saved before our forecast was issued (T+10): issue time and {valid time: [(name, lat, lon)]}."""
    best = None
    for page in sorted(root.glob("*/[0-9][0-9][0-9][0-9].html")):
        dongset = page.with_name(page.stem + "_dongset.json")
        if fetch_time(page) > anchor + timedelta(minutes=10) or not dongset.exists():
            continue
        issued = page_frames(page)[1]
        if issued is not None and (best is None or issued >= best[0]):
            best = (issued, dongset)
    if best is None:
        return None
    out = {}
    for key, items in json.loads(best[1].read_text(encoding="utf-8")).items():
        pts = []
        for item in items:
            name, province, pos = (x.strip() for x in item.rsplit(",", 2))
            lat, lon = (float(v) for v in pos.split("-"))
            pts.append((f"{name}, {province}", lat, lon))
        out[datetime.strptime(key, "%Y%m%d%H%M")] = pts
    return best[0], out


def hymetnet_strokes(root: Path, anchor: datetime, t0: datetime, t1: datetime) -> np.ndarray | None:
    """(lat, lon) of the strokes in [t0, t1) in hymetnet's frame of (T, T+60], if a page holding it was saved: once each
    (the page repeats some records), within the iWeather feed's radius."""
    found = verify_page(root, anchor)
    if found is None:
        return None
    text = found[0].read_text(encoding="utf-8", errors="replace")
    body = re.search(rf"set\[{found[1]}\] = \[(.*?)\];", text, re.DOTALL)
    pts = set()
    for rec in re.findall(r"\{([^}]*)\}", body.group(1) if body else ""):
        d = dict(kv.split(":", 1) for kv in rec.split(","))
        if "undefined" in d.get("nam", "undefined"):
            continue
        t = datetime(int(d["nam"]), int(d["thang"]), int(d["ngay"]), int(d["gio"]), int(d["phut"]), int(d.get("giay", 0)))
        lat, lon = float(d["lat"]), float(d["lng"])
        if t0 <= t < t1 and haversine_km(FEED_LAT, FEED_LON, lat, lon) <= FEED_RADIUS_KM:
            pts.add((t, lat, lon))
    return np.array([(lat, lon) for _, lat, lon in sorted(pts)]).reshape(-1, 2)


def disc_mask(lat: np.ndarray, lon: np.ndarray, radius_km: float) -> np.ndarray:
    """Grid cells within radius_km of any of the points."""
    m = np.zeros((OUTER, OUTER), bool)
    iy, ix = cell_of(lon, lat)
    ok = (iy >= 0) & (iy < OUTER) & (ix >= 0) & (ix < OUTER)
    m[iy[ok], ix[ok]] = True
    r = int(radius_km / RES_KM)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_dilation(m, structure=(yy ** 2 + xx ** 2) <= (radius_km / RES_KM) ** 2)


def covered(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    iy, ix = cell_of(lon, lat)
    ok = (iy >= 0) & (iy < OUTER) & (ix >= 0) & (ix < OUTER)
    out = np.zeros(lat.size, bool)
    out[ok] = mask[iy[ok], ix[ok]]
    return out


def extent(geo: Geo, prob: np.ndarray, lats: list[np.ndarray], lons: list[np.ndarray]) -> tuple[float, float, float, float]:
    """The part of the map where something happens, padded (at least 1.6 degrees across)."""
    field = geo.raster.sample(prob)
    lon0, lon1, lat0, lat1 = geo.raster.extent
    rows, cols = np.nonzero(np.nan_to_num(field) >= 0.1)
    la = [lat0 + (rows + 0.5) * (lat1 - lat0) / field.shape[0]] + lats
    lo = [lon0 + (cols + 0.5) * (lon1 - lon0) / field.shape[1]] + lons
    la, lo = np.concatenate(la), np.concatenate(lo)
    if la.size == 0:
        return 105.5, 107.5, 9.7, 11.5
    cy, cx = (la.min() + la.max()) / 2, (lo.min() + lo.max()) / 2
    half = max(0.8, (la.max() - la.min()) / 2 + 0.25, (lo.max() - lo.min()) / 2 + 0.25)
    return cx - half, cx + half, cy - half, cy + half


def base(ax: plt.Axes, geo: Geo, ext: tuple[float, float, float, float]) -> None:
    for ring in geo.outlines:
        ax.plot(ring[:, 0], ring[:, 1], color=RULE, lw=0.6, zorder=2)
    for ring in geo.coast:
        ax.plot(ring[:, 0], ring[:, 1], color=INK_2, lw=0.8, zorder=2)
    labelled: list[tuple[float, float]] = []
    min_deg = (ext[1] - ext[0]) / 14  # no label closer than this to another (the city's communes crowd otherwise)
    for p in (c for c in geo.communes if c.notable):
        if ext[0] <= p.lon <= ext[1] and ext[2] <= p.lat <= ext[3]:
            ax.plot(p.lon, p.lat, "o", ms=4, mfc="white", mec=INK, mew=0.8, zorder=6)
            if all(np.hypot(p.lon - x, p.lat - y) >= min_deg for x, y in labelled):
                ax.annotate(short_name(p.name), (p.lon, p.lat), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK, zorder=6)
                labelled.append((p.lon, p.lat))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.radians((ext[2] + ext[3]) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])


def panel(work_dir: Path, anchor: datetime, geo: Geo, feed: Path) -> Path | None:
    folder = work_dir / "forecasts" / f"{anchor:%Y%m%d}" / f"{anchor:%H%M}"
    if not (folder / "forecast.json").exists():
        print(f"{anchor}: no forecast")
        return None
    fc = Forecast.read(folder)
    k = fc.level_index
    w = fc.windows[k]
    t0, t1 = anchor + timedelta(minutes=10), anchor + timedelta(minutes=w.end_min)
    hm = hymetnet_warnings(work_dir / "hymetnet", anchor)
    hm_end = min(t1, max(hm[1]) if hm and hm[1] else t1)
    common = (t0, hm_end)  # the window both cover; hymetnet's warnings may end before ours

    s = load_strokes(feed, t0, t1)
    st = s.time.astype("datetime64[s]").astype(datetime)
    in_common = np.array([common[0] < t <= common[1] for t in st], bool)
    hm_pts = [(n, la, lo) for vt, pts in (hm[1].items() if hm else []) if t0 < vt <= t1 for n, la, lo in pts]
    hm_lat = np.array([p[1] for p in hm_pts])
    hm_lon = np.array([p[2] for p in hm_pts])
    hm_obs = hymetnet_strokes(work_dir / "hymetnet", anchor, t0, t1)

    ours = fc.prob[k] >= ALERT
    theirs = disc_mask(hm_lat, hm_lon, RADIUS_KM) if hm_pts else np.zeros_like(ours)
    n = int(in_common.sum())
    hit_ours = int(covered(ours, s.lat[in_common], s.lon[in_common]).sum())
    hit_ours_all = int(covered(ours, s.lat, s.lon).sum())
    hit_theirs = int(covered(theirs, s.lat[in_common], s.lon[in_common]).sum())
    cell_km2 = RES_KM ** 2

    ext = extent(geo, fc.prob[k], [s.lat, hm_lat], [s.lon, hm_lon])
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.6), facecolor="white")
    for ax in axes:
        base(ax, geo, ext)
    cmap = ListedColormap(list(MAP.prob))
    cmap.set_under((0, 0, 0, 0))
    cmap.set_bad((0, 0, 0, 0))
    ax = axes[0]
    ax.imshow(geo.raster.sample(fc.prob[k]), extent=geo.raster.extent, origin="lower", cmap=cmap, norm=BoundaryNorm(PROB_EDGES, len(MAP.prob)), zorder=1)
    lon_g = np.linspace(*geo.raster.extent[:2], geo.raster.iy.shape[1])
    lat_g = np.linspace(*geo.raster.extent[2:], geo.raster.iy.shape[0])
    ax.contour(lon_g, lat_g, np.nan_to_num(geo.raster.sample(fc.prob[k])), levels=[fc.levels.alert, fc.levels.warning],
               colors=[LEVEL_COLOR[Level.ALERT], LEVEL_COLOR[Level.WARNING]], linewidths=1.6, zorder=4)
    ax.set_title(f"Dự báo của chúng tôi · phát lúc {vn(fc.issued)}\nxác suất có sét trong {w.radius_km:g} km, {vn(t0)}–{vn(t1)}"
                 f" · đường cam ≥ {fc.levels.alert:.0%}, đỏ ≥ {fc.levels.warning:.0%}", fontsize=11, loc="left")
    ax = axes[1]
    if hm_pts:
        for ring_lat, ring_lon in zip(hm_lat, hm_lon):
            a = np.linspace(0, 2 * np.pi, 48)
            ax.fill(ring_lon + RADIUS_KM * np.sin(a) / (111.32 * np.cos(np.radians(ring_lat))), ring_lat + RADIUS_KM * np.cos(a) / 111.32,
                    color=HM_COLOR, alpha=0.08, lw=0, zorder=3)
        ax.plot(hm_lon, hm_lat, "^", ms=7, color=HM_COLOR, mec="white", mew=0.6, zorder=5)
    hm_title = (f"Cảnh báo hymetnet có lúc {vn(fc.issued)} · phát lúc {vn(hm[0])}\nđiểm cảnh báo hiệu lực {vn(t0)}–{vn(hm_end)}"
                f" ({len(hm_pts)} điểm) · vòng mờ: 10 km quanh điểm") if hm else "Cảnh báo hymetnet: không có số liệu lưu"
    ax.set_title(hm_title, fontsize=11, loc="left")
    ax = axes[2]
    if hm_obs is not None and hm_obs.size:
        ax.plot(hm_obs[:, 1], hm_obs[:, 0], "o", ms=9, mfc="none", mec=HM_COLOR, mew=1.0, zorder=4, label=f"hymetnet ({len(hm_obs)})")
    ax.plot(s.lon, s.lat, "x", ms=6, color=MAP.verify, mew=1.4, zorder=5, label=f"iWeather ({len(s)})")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.set_title(f"Sét thực tế {vn(t0)}–{vn(t1)}\n✕ iWeather" + (" · ○ hymetnet (cùng mạng định vị)" if hm_obs is not None else
                                                              " · hymetnet: khung này chưa lưu"), fontsize=11, loc="left")
    # the same strokes over both warnings, for the eye
    for ax in axes[:2]:
        ax.plot(s.lon, s.lat, "x", ms=4, color=MAP.verify, mew=0.9, alpha=0.7, zorder=5)

    def share(h: int) -> str:
        return f"{h}/{n} cú sét ({h / n:.0%})" if n else "không có sét"

    def km2(m: np.ndarray) -> str:
        return f"{int(m.sum() * cell_km2):,}".replace(",", ".")

    full = f"{hit_ours_all}/{len(s)} ({hit_ours_all / len(s):.0%})" if len(s) else "không có sét"
    axes[0].set_xlabel(f"{vn(common[0])}–{vn(common[1])}: trong vùng ≥ {ALERT:.0%} {share(hit_ours)}\n"
                       f"cả khung {vn(t0)}–{vn(t1)}: {full} · vùng ≥ {ALERT:.0%}: {km2(ours)} km²", fontsize=11, loc="left")
    axes[1].set_xlabel(f"{vn(common[0])}–{vn(common[1])}: trong 10 km quanh điểm {share(hit_theirs)}\n"
                       f"sau {vn(common[1])}: không có cảnh báo hymetnet phát trước {vn(fc.issued)} · vùng: {km2(theirs)} km²"
                       if common[1] < t1 else f"trong 10 km quanh điểm: {share(hit_theirs)} · vùng: {km2(theirs)} km²", fontsize=11, loc="left")
    axes[2].set_xlabel(f"{len(s)} cú sét iWeather trong khung, {int(s.in_cloud.sum())} trong mây", fontsize=11, loc="left")
    fig.suptitle(f"Dự báo sét lúc {vn(anchor)} ngày {(anchor + VN):%d/%m/%Y} (giờ VN, radar lúc {vn(anchor)}) — so với cảnh báo hymetnet và sét thực tế",
                 fontsize=14, x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.01, f"Chấm điểm trên khung cả hai cùng phủ ({vn(common[0])}–{vn(common[1])}), với sét iWeather; sét ✕ được vẽ trên cả ba bản đồ. "
             "Cảnh báo hymetnet là các điểm (xã) không có diện tích: tính vùng 10 km quanh điểm, như định nghĩa dự báo của chúng tôi.\n"
             "Nguồn: hymetnet.gov.vn/lightningmaps và hymetnet.gov.vn/dongset, lưu mỗi 5 phút; iWeather (định vị sét).", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = work_dir / "moments" / f"{anchor:%Y%m%d}" / f"{anchor:%H%M}_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}: ours {hit_ours}/{n}, hymetnet {hit_theirs}/{n} ({vn(common[0])}-{vn(common[1])})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--lightning-feed", type=Path, default=Path("data/lightning/feed"))
    ap.add_argument("anchors", nargs="*", type=datetime.fromisoformat)
    args = ap.parse_args()
    anchors = args.anchors
    if not anchors:
        with (args.work_dir / "moments" / "moments.csv").open() as f:
            anchors = [datetime.fromisoformat(r["anchor"]) for r in csv.DictReader(f) if r["flagged"] == "1"]
    geo = Geo.load(DEFAULT_BOUNDARIES, DEFAULT_NOTABLE)
    for a in anchors:
        panel(args.work_dir, a, geo, args.lightning_feed)


if __name__ == "__main__":
    main()
