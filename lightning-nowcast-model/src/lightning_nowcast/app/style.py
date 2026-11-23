"""Display names, local time and colors of the app (one place for every visual constant)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..grid import Source
from ..product import Level

ICT = timedelta(hours=7)  # Vietnam time, as the duty schedule and the bulletins use
STALE_AFTER = timedelta(minutes=15)  # no newer forecast than this: the feed is stale

SOURCE_NAMES = {Source.RADAR120: "Radar 120 km", Source.RADAR300: "Radar 300 km", Source.SATELLITE: "Vệ tinh", Source.LIGHTNING: "Định vị sét"}
LEVEL_TEXT = {Level.NONE: "Không", Level.ALERT: "Theo dõi", Level.WARNING: "Cảnh báo"}
LEVEL_ICON = {Level.NONE: "–", Level.ALERT: "●", Level.WARNING: "▲"}
LEVEL_RANK = {Level.NONE: 0, Level.ALERT: 1, Level.WARNING: 2}

# surfaces and ink (text never takes a data color)
SURFACE = "#fcfcfb"
PANEL = "#f4f3f0"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
RULE = "#d9d8d3"
# status colors, reserved for the levels and always shown with icon + label
LEVEL_COLOR = {Level.NONE: None, Level.ALERT: "#fab219", Level.WARNING: "#d03b3b"}
STALE_COLOR = "#d03b3b"
GOOD = "#0ca30c"

# probability steps (their colors: MAP.prob); below 10% is not drawn. Seven steps is the most a ramp keeps visibly apart
PROB_EDGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001]
# radar CMAX: the operators' IRIS color table (steps and colors as its legend), so the radar reads as on their display
DBZ_EDGES = [2, 7, 12, 18, 21, 23, 28, 34, 37, 39, 44, 50, 53, 55, 60, 66, 100]
DBZ_COLORS = ["#0085e2", "#00b7b9", "#009485", "#008d4b", "#00ac2f", "#00d518", "#13f418", "#97ff21",
              "#fff20e", "#ffd000", "#ffa900", "#ff7a00", "#f83500", "#d5071e", "#df00a1", "#ff00ff"]
DBZ_CONTOUR = 35.0  # the draft procedure's 35-40 dBZ cue
# strokes: shape = kind (+ ground, dot in-cloud), ink = age
STROKE_AGES = [(10.0, INK, "≤ 10 phút"), (30.0, INK_2, "10–30 phút"), (60.0, "#a3a29c", "30–60 phút")]


@dataclass(frozen=True)
class MapTheme:
    """The map's colors (the panels keep the light UI): a quiet base (land and sea filled, as the operators' IRIS display)
    under one saturated family for the data, labels on top."""
    sea: str
    land: str        # Vietnam
    foreign: str     # the neighbouring countries
    outline: str     # province boundaries and borders
    coast: str
    ink: str         # labels, communes, the radar
    halo: str        # behind labels
    muted: str       # range rings
    prob: tuple[str, ...]  # the probability steps of PROB_EDGES (also the place table's cells)
    verify: str
    strokes: tuple[str, str, str]  # by age, as STROKE_AGES
    contours: tuple[str, str]      # the alert / warning lines: ink, dashed / solid (the level colors would vanish in the bands)
    contour_styles: tuple[str, str]
    contour_widths: tuple[float, float]


# "IRIS soft": IRIS's khaki land and blue sea, toned down; probability pink -> plum, the complement of khaki
# (OKLab L monotone, adjacent steps >= 0.06); teal verification
MAP = MapTheme(sea="#b9c7d6", land="#ddd5b5", foreign="#ebe6da", outline="#9c9480", coast="#4a463c", ink="#1f1d1a",
               halo="#f7f4ea", muted="#7d776a",
               prob=("#f7cde6", "#eda3d4", "#dd73bf", "#c14aa7", "#9b2d8e", "#701b72", "#470f4f"),
               verify="#0f766e", strokes=("#1f1d1a", "#4a463c", "#7d776a"),
               contours=("#1f1d1a", "#1f1d1a"), contour_styles=("--", "-"), contour_widths=(0.9, 1.5))
MAP_RADIUS_KM = 300.0  # the map shows the radar's 300 km range (outside it the forecast has no radar)


def local(t: datetime) -> datetime:
    return t + ICT


def hhmm(t: datetime) -> str:
    """Local clock time of a UTC time."""
    return f"{local(t):%H:%M}"


def day(t: datetime) -> str:
    return f"{local(t):%d/%m/%Y}"


def prob_color(p: float) -> str | None:
    """The place table's cell color: the map's."""
    for lo, hi, c in zip(PROB_EDGES[:-1], PROB_EDGES[1:], MAP.prob):
        if lo <= p < hi:
            return c
    return None


def ink_on(hex_color: str) -> str:
    """Readable text color on a filled background."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


def short_name(name: str) -> str:
    """'Phường Bến Thành' -> 'P. Bến Thành', 'Xã Củ Chi' -> 'X. Củ Chi' (map labels)."""
    for long, short in (("Phường ", "P. "), ("Xã ", "X. "), ("Đặc khu ", "ĐK. ")):
        if name.startswith(long):
            return short + name[len(long):]
    return name
