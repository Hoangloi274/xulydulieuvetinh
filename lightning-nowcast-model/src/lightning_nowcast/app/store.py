"""The forecasts of a work directory: the list of issues, loading with a small cache, and the strokes after an issue."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..product import Forecast

STEP = timedelta(minutes=10)


@dataclass(frozen=True)
class Issue:
    """One forecast on disk (complete: its forecast.json exists)."""
    anchor: datetime
    folder: Path


@dataclass(frozen=True)
class Strokes:
    time: np.ndarray  # datetime64[s] UTC
    lat: np.ndarray
    lon: np.ndarray
    ground: np.ndarray

    @classmethod
    def empty(cls) -> Strokes:
        none = np.zeros(0, np.float32)
        return cls(np.zeros(0, "datetime64[s]"), none, none, np.zeros(0, bool))


@dataclass(frozen=True)
class Verification:
    """Strokes in a window after its forecast was issued; `until` = how far the later forecasts cover it."""
    strokes: Strokes
    until: datetime


ISSUE_DELAY = timedelta(minutes=9)  # a forecast is out ~9 min after its radar time


class ForecastStore:
    def __init__(self, work_dir: Path, cache: int = 24, until: datetime | None = None) -> None:
        self.root = work_dir / "forecasts"
        self.until = until  # replays: only the forecasts issued by then (UTC)
        self._cache: OrderedDict[Path, Forecast] = OrderedDict()
        self._size = cache

    def issues(self, hours: float = 24.0) -> list[Issue]:
        """Forecasts of the last `hours` before the newest one, oldest first."""
        found = []
        for folder in self.root.glob("[0-9]" * 8 + "/" + "[0-9]" * 4):
            if (folder / "forecast.json").exists():
                found.append(Issue(datetime.strptime(folder.parent.name + folder.name, "%Y%m%d%H%M"), folder))
        found.sort(key=lambda i: i.anchor)
        if self.until is not None:
            found = [i for i in found if i.anchor + ISSUE_DELAY <= self.until]
        if found:
            first = found[-1].anchor - timedelta(hours=hours)
            found = [i for i in found if i.anchor >= first]
        return found

    def load(self, issue: Issue) -> Forecast:
        if issue.folder in self._cache:
            self._cache.move_to_end(issue.folder)
            return self._cache[issue.folder]
        fc = Forecast.read(issue.folder)
        self._cache[issue.folder] = fc
        while len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return fc

    def find(self, anchor: datetime) -> Issue | None:
        folder = self.root / f"{anchor:%Y%m%d}" / f"{anchor:%H%M}"
        return Issue(anchor, folder) if (folder / "forecast.json").exists() else None

    def previous(self, anchor: datetime) -> Forecast | None:
        issue = self.find(anchor - STEP)
        return self.load(issue) if issue else None

    def verification(self, anchor: datetime, end_min: int) -> Verification:
        """Strokes in the window (T+10, T+end_min] of the forecast at `anchor`, from the observed strokes that the
        later forecasts kept (each holds the hour up to its T+5)."""
        t0, t1 = anchor + timedelta(minutes=10), anchor + timedelta(minutes=end_min)
        parts, until = [], t0
        a = anchor
        while a <= t1 + timedelta(minutes=55):
            if (issue := self.find(a)) is not None:
                o = self.load(issue).observed
                # to the ms: whole seconds would split one stroke kept by two forecasts across a second boundary
                t = np.datetime64(a, "ms") + np.round(o.stroke_min.astype(np.float64) * 60_000).astype("timedelta64[ms]")
                parts.append((t, o.stroke_lat, o.stroke_lon, o.stroke_ground))
                until = max(until, a + timedelta(minutes=5))
            a += STEP
        if not parts:
            return Verification(Strokes.empty(), t0)
        t, lat, lon, ground = (np.concatenate(x) for x in zip(*parts))
        keep = (t > np.datetime64(t0, "s")) & (t <= np.datetime64(t1, "s"))
        # one stroke per 0.1 s and position (the float32 minutes of different forecasts differ by < 1 ms)
        _, first = np.unique(np.stack([np.round(t[keep].astype(np.int64) / 100), np.round(lat[keep] * 1e4), np.round(lon[keep] * 1e4)], 1),
                             axis=0, return_index=True)
        sel = np.flatnonzero(keep)[first]
        return Verification(Strokes(t[sel].astype("datetime64[s]"), lat[sel], lon[sel], ground[sel]), min(until, t1))
