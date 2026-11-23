"""uv run python -m lightning_nowcast.app --work-dir live [--hours 24] [--issue 2026-06-01T06:00] [--screenshot out.png [--window 60] [--radar]]"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from ..regions import DEFAULT_BOUNDARIES, DEFAULT_NOTABLE


def main() -> None:
    ap = argparse.ArgumentParser(description="Operator app for the live lightning forecasts")
    ap.add_argument("--work-dir", type=Path, required=True, help="The live daemon's --work-dir (local or a shared folder)")
    ap.add_argument("--boundaries-dir", type=Path, default=DEFAULT_BOUNDARIES)
    ap.add_argument("--notable", type=Path, default=DEFAULT_NOTABLE, help="The communes labelled and listed first (CSV)")
    ap.add_argument("--hours", type=float, default=24.0, help="History kept on the slider")
    ap.add_argument("--issue", type=datetime.fromisoformat, default=None, help="Open this forecast (radar time T, UTC), e.g. 2026-06-01T06:00")
    ap.add_argument("--screenshot", type=Path, default=None, help="Render the newest forecast to an image and exit")
    ap.add_argument("--window", type=int, default=None, metavar="MIN", help="Screenshot: the window ending at T+MIN (default: the levels' window)")
    ap.add_argument("--radar", action="store_true", help="Screenshot: radar CMAX as the base layer")
    ap.add_argument("--clock", type=datetime.fromisoformat, default=None,
                    help="Replays: show the app as at this time (UTC): later forecasts hidden, ages counted from it")
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 950), metavar=("W", "H"))
    args = ap.parse_args()

    import logging

    import matplotlib
    matplotlib.use("QtAgg")
    logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)  # zooming with a fixed aspect logs a harmless notice
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .geo import Geo
    from .store import ForecastStore
    from .window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    if args.clock is not None:
        from . import window as window_module
        window_module.CLOCK = args.clock
    window = MainWindow(ForecastStore(args.work_dir, until=args.clock), Geo.load(args.boundaries_dir, args.notable), args.hours, args.issue)
    window.resize(*args.size)
    window.show()
    if window.fc is not None and args.window is not None:
        window.lead.setValue(next(i for i, w in enumerate(window.fc.windows) if w.end_min == args.window))
    if args.radar:
        from .mapview import Base
        window.base.setCurrentIndex(list(Base).index(Base.RADAR))
    if args.screenshot:
        def shoot() -> None:
            window.canvas.draw()
            window.grab().save(str(args.screenshot))
            app.quit()
        QTimer.singleShot(1500, shoot)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
