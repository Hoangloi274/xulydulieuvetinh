"""Add the commune warnings to forecasts written before the communes (they carry places / provinces / regions), from
their stored probabilities, exactly as the daemon now computes them (live.Areas). The other fields are kept.

    uv run python scripts/reissue_communes.py --work-dir live [--domain-km 290]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lightning_nowcast.live import Areas
from lightning_nowcast.product import Levels
from lightning_nowcast.regions import DEFAULT_BOUNDARIES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--boundaries-dir", type=Path, default=DEFAULT_BOUNDARIES)
    ap.add_argument("--domain-km", type=float, default=290.0, help="As the model's DataConfig.domain_km (290 km)")
    args = ap.parse_args()
    areas = Areas(args.boundaries_dir, args.domain_km)
    done = skipped = 0
    for fj in sorted(args.work_dir.glob("forecasts/*/*/forecast.json")):
        d = json.loads(fj.read_text(encoding="utf-8"))
        if "communes" in d:
            skipped += 1
            continue
        levels = Levels(**d["levels"])
        k = next(i for i, w in enumerate(d["windows"]) if w["end_min"] == levels.window_min and w["radius_km"] == levels.radius_km)
        with np.load(fj.parent / "prob.npz") as z:
            prob = z["prob"].astype(np.float32)
        d["communes"] = [{"code": a.code, "name": a.name, "province": a.province, "probability": list(a.probability), "level": a.level.value}
                         for a in areas.warnings(prob, levels, k)]
        tmp = fj.with_name(fj.name + ".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(fj)
        done += 1
    print(f"{done} forecasts given communes, {skipped} already had them")


if __name__ == "__main__":
    main()
