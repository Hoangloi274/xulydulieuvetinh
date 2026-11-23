"""Run the model over packed inputs once and keep its probability fields, for verification that sweeps thresholds.

    uv run python -m lightning_nowcast.forecast_cache --ckpt models/lightning_nowcast.pt \\
        --packed data/stores/packed_test --start 2026-05-29T00:00 --end 2026-06-30T23:50 --out runs/cache_2026.npz

Output (npz): anchors (N,) datetime64[m] UTC, prob (N, n_windows, 320, 320) float16 (calibrated if the checkpoint
is), windows_min, radii_km and thresholds, one per window. Anchors without complete inputs are left out.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    DataConfig,
    PackedStores,
    Samples,
    decode_batch,
    valid_anchors,
    worker_context,
)
from .grid import OUTER
from .model import load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--packed", type=Path, required=True)
    ap.add_argument("--start", type=datetime.fromisoformat, required=True, help="First anchor (UTC)")
    ap.add_argument("--end", type=datetime.fromisoformat, required=True, help="Last anchor (UTC)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (~0.7 GB of host RAM each)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = DataConfig.from_checkpoint(ckpt)
    model = load_model(ckpt, args.device)
    cal = ckpt.get("calibration")
    a = torch.tensor(cal["a"], device=args.device).view(1, -1, 1, 1) if cal else None
    b = torch.tensor(cal["b"], device=args.device).view(1, -1, 1, 1) if cal else None
    stores = PackedStores(args.packed)
    anchors = valid_anchors(stores, cfg, np.datetime64(args.start, "m"), np.datetime64(args.end, "m"))
    dl = DataLoader(Samples(stores, anchors, cfg), batch_size=args.batch, num_workers=args.workers,
                    multiprocessing_context=worker_context(args.workers))
    prob = np.zeros((anchors.size, len(cfg.names), OUTER, OUTER), np.float16)
    t0, n = time.time(), 0
    with torch.inference_mode():
        for batch in dl:
            d = decode_batch(batch, cfg, args.device)
            logit, _ = model({k: d[k] for k in cfg.inputs}, d["static"])
            if a is not None:
                logit = a * logit + b
            prob[n:n + logit.shape[0]] = torch.sigmoid(logit).cpu().numpy().astype(np.float16)
            n += logit.shape[0]
    thr = [float(ckpt.get("product_thresholds", {}).get(f"w{nm}", 0.5)) for nm in cfg.names]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, anchors=stores.slot_time(anchors).astype("datetime64[m]"), prob=prob, windows_min=np.array(cfg.windows_min),
             radii_km=np.array(cfg.radii_km), thresholds=np.array(thr), ckpt=str(args.ckpt))
    print(f"wrote {args.out}: {anchors.size} anchors in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
