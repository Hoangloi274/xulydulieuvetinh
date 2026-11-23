"""The forecast product as the live daemon writes it and the operator app reads it (no torch needed).

<work-dir>/forecasts/YYYYmmdd/HHMM/   (anchor time, UTC)
    forecast.json   times, windows, levels, input health, commune warnings
    prob.npz        prob (n_windows, 320, 320) float16, rows south -> north, 2 km cells
    observed.npz    radar CMAX at T and the strokes of the hour up to T+5 (what the model saw last)
<work-dir>/latest.json   forecast.json of the newest forecast, plus its "dir"

forecast.json is written last (atomically), so a directory with it is complete. Its communes are those reaching
LISTED_FROM in some window (a commune not listed is below it in every window); forecasts written before the communes
(places, provinces, regions) read with no communes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .grid import OUTER, Source

LISTED_FROM = 0.05  # forecast.json lists the communes reaching this in some window


class Level(Enum):
    NONE = "none"
    ALERT = "alert"      # theo dõi
    WARNING = "warning"  # cảnh báo


@dataclass(frozen=True)
class Levels:
    """Operating thresholds, applied to the window `window_min` / `radius_km` (chosen on 2026 period 1)."""
    alert: float = 0.30
    warning: float = 0.40
    window_min: int = 60
    radius_km: float = 10.0

    def of(self, p: float) -> Level:
        return Level.WARNING if p >= self.warning else Level.ALERT if p >= self.alert else Level.NONE


@dataclass(frozen=True)
class Window:
    end_min: int       # window (T+10, T+end_min]
    radius_km: float


@dataclass(frozen=True)
class SourceHealth:
    frames: int  # frames present among those the model reads
    needed: int

    @property
    def complete(self) -> bool:
        return self.frames == self.needed


@dataclass(frozen=True)
class AreaWarning:
    """A ward or commune: the highest probability over its grid cells (lightning within the window's radius of some
    point of it), per window."""
    code: str
    name: str
    province: str
    probability: tuple[float, ...]
    level: Level  # from the levels' window


@dataclass(frozen=True)
class Observed:
    """What the forecaster saw last, kept with the forecast for display (observed.npz)."""
    cmax: np.ndarray  # (320, 320) float16 dBZ at slot T: the 120 km volume where it reaches, else the 300 km one; NaN = none
    stroke_min: np.ndarray  # stroke times, minutes after T (the hour up to T+5)
    stroke_lat: np.ndarray
    stroke_lon: np.ndarray
    stroke_ground: np.ndarray  # cloud-to-ground (else in-cloud)

    @classmethod
    def empty(cls) -> Observed:
        none = np.zeros(0, np.float32)
        return cls(np.full((OUTER, OUTER), np.nan, np.float16), none, none, none, np.zeros(0, bool))


@dataclass(frozen=True)
class Forecast:
    anchor: datetime
    issued: datetime
    model: str  # checkpoint name @ the first 8 hex digits of its sha256
    windows: tuple[Window, ...]
    levels: Levels
    prob: np.ndarray = field(repr=False)  # (len(windows), 320, 320) float32, rows south -> north, 2 km cells
    observed: Observed = field(repr=False)
    health: dict[Source, SourceHealth]
    communes: tuple[AreaWarning, ...]  # those reaching LISTED_FROM in some window, in bulletin order

    @property
    def level_index(self) -> int:
        """The window the levels apply to."""
        return next(i for i, w in enumerate(self.windows) if w.end_min == self.levels.window_min and w.radius_km == self.levels.radius_km)

    def write(self, root: Path) -> Path:
        out = root / "forecasts" / f"{self.anchor:%Y%m%d}" / f"{self.anchor:%H%M}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "prob.npz", prob=self.prob.astype(np.float16))
        o = self.observed
        np.savez_compressed(out / "observed.npz", cmax=o.cmax, stroke_min=o.stroke_min, stroke_lat=o.stroke_lat,
                            stroke_lon=o.stroke_lon, stroke_ground=o.stroke_ground)
        meta = {k: v for k, v in asdict(self).items() if k not in ("prob", "observed")}

        def plain(v: Any) -> str:
            return v.value if isinstance(v, Enum) else v.isoformat() if isinstance(v, datetime) else str(v)

        text = json.dumps(meta, default=plain, ensure_ascii=False, indent=1)
        _write_atomic(out / "forecast.json", text)
        _write_atomic(root / "latest.json", json.dumps({"dir": str(out.relative_to(root)), **json.loads(text)}, ensure_ascii=False, indent=1))
        return out

    @classmethod
    def read(cls, folder: Path) -> Forecast:
        d = json.loads((folder / "forecast.json").read_text(encoding="utf-8"))
        with np.load(folder / "prob.npz") as z:
            prob = z["prob"].astype(np.float32)
        observed = Observed.empty()
        if (folder / "observed.npz").exists():
            with np.load(folder / "observed.npz") as z:
                observed = Observed(z["cmax"], z["stroke_min"], z["stroke_lat"], z["stroke_lon"], z["stroke_ground"])

        return cls(anchor=datetime.fromisoformat(d["anchor"]), issued=datetime.fromisoformat(d["issued"]), model=d["model"],
                   windows=tuple(Window(**w) for w in d["windows"]), levels=Levels(**d["levels"]), prob=prob, observed=observed,
                   health={Source(k): SourceHealth(**v) for k, v in d["health"].items()},
                   communes=tuple(AreaWarning(a["code"], a["name"], a["province"], tuple(a["probability"]), Level(a["level"]))
                                  for a in d.get("communes", ())))


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
