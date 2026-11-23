"""The map: probability (or radar) bands, level contours, communes by level, strokes, rings; zoom and hover."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from PySide6.QtCore import Signal

from ..grid import CENTER_LAT, grid_lonlat, range_km_map
from ..product import Forecast, Level
from ..regions import Commune
from .geo import Geo, cell_of
from .store import Verification
from .style import (
    DBZ_COLORS,
    DBZ_CONTOUR,
    DBZ_EDGES,
    LEVEL_COLOR,
    LEVEL_RANK,
    LEVEL_TEXT,
    MAP,
    MAP_RADIUS_KM,
    PROB_EDGES,
    STROKE_AGES,
    short_name,
)

ZOOMED_DEG = 0.8  # a view narrower than this names any commune, not only the notable ones


class Base(Enum):
    PROBABILITY = "Xác suất sét"
    RADAR = "Radar CMAX"


@dataclass
class Layers:
    base: Base = Base.PROBABILITY
    contours: bool = True       # probability contours at the alert and warning levels
    communes: bool = True       # communes at alert / warning outlined in their level's color; notable ones dotted
    strokes: bool = True        # strokes of the hour up to T+5
    verify: bool = True         # strokes that then fell in the window (past forecasts)
    radar_contour: bool = False  # the 35 dBZ outline
    rings: bool = True


def _cmap(colors: list[str]) -> ListedColormap:
    cmap = ListedColormap(colors)
    cmap.set_under((0, 0, 0, 0))
    cmap.set_bad((0, 0, 0, 0))
    return cmap


class MapCanvas(FigureCanvasQTAgg):
    hovered = Signal(str)

    def __init__(self, geo: Geo) -> None:
        self.fig = Figure(facecolor=MAP.sea)
        super().__init__(self.fig)
        self.geo = geo
        self.lon, self.lat = grid_lonlat()
        ax = self.ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.set_facecolor(MAP.sea)
        ax.set_aspect(1 / np.cos(np.deg2rad(CENTER_LAT)), adjustable="datalim")  # zoom and pan change the view, not the box
        ext = geo.raster.extent
        # the view and everything drawn stop at the radar's range
        edge = geo.rings.get(int(MAP_RADIUS_KM))
        self.clip = Polygon(edge, closed=True, transform=ax.transData) if edge is not None else None
        pad = 0.02 * (edge[:, 0].max() - edge[:, 0].min()) if edge is not None else 0.0
        self.home = (edge[:, 0].min() - pad, edge[:, 0].max() + pad, edge[:, 1].min() - pad, edge[:, 1].max() + pad) if edge is not None else ext
        self.grid_outside = range_km_map() > MAP_RADIUS_KM
        self.prob_cmap, self.prob_norm = _cmap(list(MAP.prob)), BoundaryNorm(PROB_EDGES, len(MAP.prob))
        self.dbz_cmap, self.dbz_norm = _cmap(DBZ_COLORS), BoundaryNorm(DBZ_EDGES, len(DBZ_COLORS))
        lines = [PolyCollection(geo.foreign, facecolors=MAP.foreign, edgecolors=MAP.outline, linewidths=0.6, zorder=0),
                 PolyCollection(geo.coast, facecolors=MAP.land, edgecolors="none", zorder=0),
                 LineCollection(geo.outlines, colors=MAP.outline, linewidths=0.6, zorder=4),
                 LineCollection(geo.coast, colors=MAP.coast, linewidths=0.8, zorder=4)]
        for c in lines:
            ax.add_collection(c)
        self.notable = [c for c in geo.communes if c.notable]
        self.commune_dots = ax.scatter([c.lon for c in self.notable], [c.lat for c in self.notable], s=26, facecolors=MAP.halo,
                                       edgecolors=MAP.ink, linewidths=0.9, zorder=8)
        self.level_lines = LineCollection([], linewidths=1.3, zorder=6)  # communes at a level, in its color
        ax.add_collection(self.level_lines)
        halo = [pe.withStroke(linewidth=2.2, foreground=MAP.halo)]  # labels sit above the bands, readable on any of them
        self.commune_labels = [ax.text(c.lon, c.lat, f" {short_name(c.name)}", fontsize=7, color=MAP.ink, va="center", zorder=9,
                                       path_effects=halo, visible=False) for c in geo.communes]
        self.commune_levels = [Level.NONE] * len(geo.communes)
        self.commune_prob = [0.0] * len(geo.communes)
        self.show_communes = True
        ax.callbacks.connect("xlim_changed", lambda _: self._update_labels())
        self.ring_lines = [ax.plot(r[:, 0], r[:, 1], color=MAP.muted, lw=0.8, ls=(0, (2, 3)), zorder=3)[0] for r in geo.rings.values()]
        self.ring_labels = [ax.text(r[0, 0], r[0, 1], f" {km} km", color=MAP.muted, fontsize=7, va="bottom", zorder=3) for km, r in geo.rings.items()]
        ax.plot(*geo.center, marker="^", color=MAP.ink, ms=6, zorder=9)
        ax.text(geo.center[0], geo.center[1], "  Nhà Bè", color=MAP.ink, fontsize=7, va="center", zorder=9, path_effects=halo)
        self.ground = ax.scatter([], [], marker="+", s=46, linewidths=1.1, zorder=7)
        self.cloud = ax.scatter([], [], marker=".", s=10, linewidths=0, zorder=7)
        self.verify = ax.scatter([], [], marker="x", s=26, linewidths=1.0, color=MAP.verify, zorder=7)
        if self.clip is not None:
            for artist in (*lines, self.level_lines, self.ground, self.cloud, self.verify):
                artist.set_clip_path(self.clip)
        self.highlight = LineCollection([], colors=MAP.ink, linewidths=2.4, zorder=8)
        ax.add_collection(self.highlight)
        self.contours: list = []
        self.legend_ax = self.fig.add_axes((0.015, 0.03, 0.34, 0.022))
        self.legend_ax.set_facecolor(MAP.sea)
        ax.set_xlim(self.home[0], self.home[1])
        ax.set_ylim(self.home[2], self.home[3])
        self.fc: Forecast | None = None
        self.k = 0
        self._drag: tuple[float, float] | None = None
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", lambda e: setattr(self, "_drag", None))

    # ------------------------------------------------------------------------------------------- drawing
    def show(self, fc: Forecast, k: int, layers: Layers, verification: Verification | None) -> None:
        self.fc, self.k = fc, k
        ax = self.ax
        for c in self.contours:
            c.remove()
        self.contours = []
        # filled bands on the forecast grid itself (smooth edges); a raster of 2 km cells looks blocky at this scale
        field = fc.prob[k].astype(np.float32) if layers.base == Base.PROBABILITY else fc.observed.cmax.astype(np.float32)
        field = np.where(self.grid_outside, np.nan, field)
        cmap, norm, edges = ((self.prob_cmap, self.prob_norm, PROB_EDGES) if layers.base == Base.PROBABILITY
                             else (self.dbz_cmap, self.dbz_norm, DBZ_EDGES))
        self.contours.append(ax.contourf(self.lon, self.lat, np.ma.masked_invalid(field), levels=edges, cmap=cmap, norm=norm, zorder=1))
        self._legend(layers.base, fc)
        if layers.contours:
            self.contours.append(ax.contour(self.lon, self.lat, fc.prob[k], levels=[fc.levels.alert, fc.levels.warning],
                                            colors=list(MAP.contours), linestyles=list(MAP.contour_styles),
                                            linewidths=list(MAP.contour_widths), zorder=5))
        if layers.radar_contour:
            self.contours.append(ax.contour(self.lon, self.lat, np.ma.masked_invalid(fc.observed.cmax.astype(np.float32)), levels=[DBZ_CONTOUR],
                                            colors=[MAP.ink], linewidths=0.8, linestyles="--", zorder=5))
        # communes at a level (of the levels' window): outlined in its color; notable ones dotted in it
        by_code = {a.code: a for a in fc.communes}
        found = [by_code.get(c.code) for c in self.geo.communes]
        self.commune_levels = [a.level if a else Level.NONE for a in found]
        self.commune_prob = [a.probability[fc.level_index] if a else 0.0 for a in found]
        segs, cols = [], []
        for c, lv in zip(self.geo.communes, self.commune_levels):
            if lv != Level.NONE:
                segs += list(c.rings)
                cols += [LEVEL_COLOR[lv]] * len(c.rings)
        self.level_lines.set_segments(segs if layers.communes else [])
        self.level_lines.set_colors(cols if layers.communes else [])
        level_of = dict(zip((c.code for c in self.geo.communes), self.commune_levels))
        self.commune_dots.set_visible(layers.communes)
        self.commune_dots.set_facecolors([to_rgba(LEVEL_COLOR[level_of[c.code]] or MAP.halo) for c in self.notable])
        self.show_communes = layers.communes
        self._update_labels()
        o = fc.observed
        age = 5.0 - o.stroke_min  # minutes before T+5, the newest stroke the model saw
        colors = np.array([to_rgba(next((c for (lim, _, _), c in zip(STROKE_AGES, MAP.strokes) if a <= lim), MAP.strokes[-1])) for a in age]).reshape(-1, 4)
        for artist, sel in ((self.ground, o.stroke_ground), (self.cloud, ~o.stroke_ground)):
            artist.set_offsets(np.column_stack([o.stroke_lon[sel], o.stroke_lat[sel]]) if layers.strokes else np.zeros((0, 2)))
            artist.set_facecolors(colors[sel] if layers.strokes else np.zeros((0, 4)))
            artist.set_edgecolors(colors[sel] if layers.strokes else np.zeros((0, 4)))
        v = verification.strokes if (verification and layers.verify) else None
        self.verify.set_offsets(np.column_stack([v.lon, v.lat]) if v is not None and v.lon.size else np.zeros((0, 2)))
        if self.clip is not None:
            for c in self.contours:
                c.set_clip_path(self.clip)
        for line in self.ring_lines + self.ring_labels:
            line.set_visible(layers.rings)
        self.draw_idle()

    def _legend(self, base: Base, fc: Forecast) -> None:
        lax = self.legend_ax
        lax.clear()
        edges, colors = (PROB_EDGES, MAP.prob) if base == Base.PROBABILITY else (DBZ_EDGES, DBZ_COLORS)
        for i, c in enumerate(colors):
            lax.add_patch(Rectangle((i, 0), 0.94, 1, facecolor=c, edgecolor="none"))
        lax.set_xlim(0, len(colors))
        lax.set_ylim(0, 1)
        lax.set_yticks([])
        every = 1 if len(colors) <= 8 else 2  # the 16 radar steps: every other label fits
        lax.set_xticks(range(0, len(colors), every))
        labels = [f"{e:.0%}" for e in edges[:-1]] if base == Base.PROBABILITY else [f"{e:g}" for e in edges[:-1]]
        labels = labels[::every]
        lax.set_xticklabels(labels, fontsize=7, color=MAP.ink)
        lax.tick_params(length=0, pad=2)
        for s in lax.spines.values():
            s.set_visible(False)
        title = "Xác suất có sét" if base == Base.PROBABILITY else "Phản hồi vô tuyến cực đại (dBZ)"
        lax.set_title(title, fontsize=7.5, color=MAP.ink, loc="left", pad=3)
        handles = [Line2D([], [], color=MAP.contours[0], ls=MAP.contour_styles[0], lw=MAP.contour_widths[0],
                          label=f"{LEVEL_TEXT[Level.ALERT]} ≥ {fc.levels.alert:.0%}"),
                   Line2D([], [], color=MAP.contours[1], ls=MAP.contour_styles[1], lw=MAP.contour_widths[1],
                          label=f"{LEVEL_TEXT[Level.WARNING]} ≥ {fc.levels.warning:.0%}"),
                   Line2D([], [], color=MAP.ink, marker="+", ls="none", ms=7, label="Sét mây–đất"),
                   Line2D([], [], color=MAP.ink, marker=".", ls="none", ms=5, label="Sét trong mây"),
                   *(Line2D([], [], color=c, marker="s", ls="none", ms=5, label=f"Sét {txt}") for (_, _, txt), c in zip(STROKE_AGES, MAP.strokes)),
                   Line2D([], [], color=MAP.verify, marker="x", ls="none", ms=6, label="Sét trong khung dự báo")]
        leg = self.ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=True, framealpha=0.92, facecolor=MAP.halo,
                             edgecolor=MAP.outline, labelcolor=MAP.ink, borderpad=0.6)
        leg.set_zorder(10)

    def _update_labels(self) -> None:
        """Name as many communes as fit: the notable ones only in a wide view, any commune once zoomed in; those at a
        level first, then by probability, then the notable; a name is left out when one already named is closer than a
        spacing that scales with the view."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        spacing, zoomed = (x1 - x0) / 14, (x1 - x0) < ZOOMED_DEG
        cs = self.geo.communes
        order = sorted(range(len(cs)), key=lambda i: (-LEVEL_RANK[self.commune_levels[i]], -self.commune_prob[i], not cs[i].notable))
        shown: list[tuple[float, float]] = []
        for i in order:
            c = cs[i]
            ok = (self.show_communes and (zoomed or c.notable) and x0 <= c.lon <= x1 and y0 <= c.lat <= y1
                  and all(math.hypot(c.lon - x, c.lat - y) >= spacing for x, y in shown))
            self.commune_labels[i].set_visible(ok)
            if ok:
                shown.append((c.lon, c.lat))

    def highlight_commune(self, commune: Commune | None) -> None:
        """Outline a commune and zoom to it."""
        if commune is None:
            self.highlight.set_segments([])
        else:
            self.highlight.set_segments(list(commune.rings))
            lon0, lon1, lat0, lat1 = self.geo.bounds(commune)
            pad = max(lon1 - lon0, lat1 - lat0, 0.15)
            self.ax.set_xlim((lon0 + lon1) / 2 - pad, (lon0 + lon1) / 2 + pad)
            self.ax.set_ylim((lat0 + lat1) / 2 - pad, (lat0 + lat1) / 2 + pad)
        self.draw_idle()

    def reset_view(self) -> None:
        self.ax.set_xlim(self.home[0], self.home[1])
        self.ax.set_ylim(self.home[2], self.home[3])
        self.draw_idle()

    # ------------------------------------------------------------------------------------------- mouse
    def _on_scroll(self, e) -> None:
        if e.xdata is None:
            return
        f = 0.8 if e.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(e.xdata - (e.xdata - x0) * f, e.xdata + (x1 - e.xdata) * f)
        self.ax.set_ylim(e.ydata - (e.ydata - y0) * f, e.ydata + (y1 - e.ydata) * f)
        self.draw_idle()

    def _on_press(self, e) -> None:
        if e.dblclick:
            self.reset_view()
        elif e.button == 1 and e.xdata is not None:
            self._drag = (e.xdata, e.ydata)

    def _on_move(self, e) -> None:
        if e.xdata is None or e.inaxes is not self.ax:
            return
        if self._drag is not None:
            dx, dy = self._drag[0] - e.xdata, self._drag[1] - e.ydata
            self.ax.set_xlim(*(v + dx for v in self.ax.get_xlim()))
            self.ax.set_ylim(*(v + dy for v in self.ax.get_ylim()))
            self.draw_idle()
            return
        self.hovered.emit(self.readout(e.xdata, e.ydata))

    def readout(self, lon: float, lat: float) -> str:
        parts = [f"{lat:.3f}°N {lon:.3f}°E"]
        commune = self.geo.commune_at(lon, lat)
        if commune is not None:
            parts.append(f"{commune.name}, {commune.province}")
        if self.fc is not None:
            iy, ix = (int(v) for v in cell_of(np.array(lon), np.array(lat)))
            if 0 <= iy < self.fc.prob.shape[1] and 0 <= ix < self.fc.prob.shape[2]:
                w = self.fc.windows[self.k]
                parts.append(f"xác suất {self.fc.prob[self.k, iy, ix]:.0%} (đến T+{w.end_min}′, {w.radius_km:g} km)")
                dbz = float(self.fc.observed.cmax[iy, ix])
                parts.append(f"{dbz:.0f} dBZ" if np.isfinite(dbz) else "không có phản hồi")
        return "  ·  ".join(parts)
