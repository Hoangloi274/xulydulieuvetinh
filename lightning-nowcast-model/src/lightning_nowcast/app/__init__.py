"""Desktop app for the operators: the live lightning forecasts of a work directory, on a map with warning tables.

    uv sync --extra cpu --extra gui
    uv run python -m lightning_nowcast.app --work-dir live

It only reads what the live daemon writes (lightning_nowcast.product), so it can run on the daemon's machine or on
any machine that sees the work directory (a shared folder). No network access is needed.
"""
