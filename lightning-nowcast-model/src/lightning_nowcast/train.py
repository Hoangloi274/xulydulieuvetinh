"""Train the lightning nowcaster from the packed stores, then calibrate it.

    uv run python -m lightning_nowcast.train --packed /root/nowcast-cache/packed --run-dir runs/train \\
        --start 2025-01-01T01:00 --end 2025-12-31T21:50

- Samples come straight from the packed monthly stores; workers read and crop uint8 codes, the GPU decodes them.
- Inputs: 120 km column radar, 300 km radar, satellite and lightning on the 320 x 320 grid, each with its own stem
  and with per-source modality dropout (the model must cope with a missing source in operation).
- Outputs: windows (T+10, T+W] within R km for each (W, R) pair: 10 km to 60 min, 25 km for 90 and 120 min.
- The kept checkpoint maximises the mean validation CSI over the windows (validation BCE starts rising a few epochs
  before CSI stops improving); training stops after --patience epochs without a better score.
- The best checkpoint is then Platt-calibrated per window on the full validation split -> <run-dir>/model.pt, with
  the thresholds of maximum CSI. log.jsonl records every epoch, including samples/s and the data-wait share.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calibration import LogitHistogram
from .dataset import (
    DataConfig,
    PackedStores,
    RunBatchSampler,
    Samples,
    decode_batch,
    split_anchors,
    valid_anchors,
    worker_context,
)
from .grid import Source
from .model import LightningNet, load_model

THRESHOLDS = np.round(np.arange(0.02, 0.91, 0.02), 2)


class WindowMetrics:
    """Pooled, masked validation scores per window, with persistence on the same pixels."""

    def __init__(self, n_windows: int) -> None:
        self.n = np.zeros(n_windows); self.pos = np.zeros(n_windows); self.brier = np.zeros(n_windows); self.bce = np.zeros(n_windows)
        self.sweep = np.zeros((len(THRESHOLDS), n_windows, 3))
        self.pers = np.zeros((n_windows, 3))

    @staticmethod
    def _counts(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> np.ndarray:
        dims = (0, 2, 3)
        return torch.stack([(pred & y & m).sum(dims), (pred & ~y & m).sum(dims), (~pred & y & m).sum(dims)], -1).double().cpu().numpy()

    @torch.no_grad()
    def update(self, logit: torch.Tensor, y: torch.Tensor, m: torch.Tensor, pers: torch.Tensor) -> None:
        p = torch.sigmoid(logit.float())
        yb, mb = y > 0.5, m > 0.5
        dims = (0, 2, 3)
        self.n += mb.sum(dims).double().cpu().numpy()
        self.pos += (yb & mb).sum(dims).double().cpu().numpy()
        self.brier += (((p - y) ** 2) * m).sum(dims).double().cpu().numpy()
        self.bce += (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * m).sum(dims).double().cpu().numpy()
        for i, t in enumerate(THRESHOLDS):
            self.sweep[i] += self._counts(p >= t, yb, mb)
        self.pers += self._counts(pers > 0.5, yb, mb)

    def result(self, names: list[str]) -> dict[str, float]:
        """Flat record for log.jsonl: w<name>_<score>."""
        out: dict[str, float] = {}
        csi = lambda c: c[..., 0] / np.maximum(c.sum(-1), 1.0)
        for k, name in enumerate(names):
            if self.n[k] == 0:
                continue
            base = self.pos[k] / self.n[k]
            sweep = csi(self.sweep[:, k])
            b = int(np.argmax(sweep))
            c = self.sweep[b, k]
            r = {"base_rate": base, "bce": self.bce[k] / self.n[k], "bss": 1 - (self.brier[k] / self.n[k]) / max(base * (1 - base), 1e-12),
                 "best_threshold": float(THRESHOLDS[b]), "best_csi": float(sweep[b]), "best_bias": float((c[0] + c[1]) / max(c[0] + c[2], 1.0)),
                 "persistence_csi": float(csi(self.pers[k]))}
            out.update({f"w{name}_{key}": float(v) for key, v in r.items()})
        out["val_bce"] = float(np.mean([out[f"w{n}_bce"] for n in names if f"w{n}_bce" in out]))
        out["score"] = float(np.mean([out[f"w{n}_best_csi"] for n in names if f"w{n}_best_csi" in out]))
        return out


def masked_bce(logit: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean BCE per window over its observed pixels, averaged over windows (each window weighs the same)."""
    loss = F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * m
    return (loss.sum((0, 2, 3)) / m.sum((0, 2, 3)).clamp(min=1.0)).mean()


def ref_loss(pred: torch.Tensor, y: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    if not bool(ok.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred.float()[ok > 0.5], y[ok > 0.5], beta=0.05)


def amp_dtype(device: str) -> torch.dtype | None:
    if not device.startswith("cuda"):
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def inputs_of(d: dict[str, torch.Tensor], cfg: DataConfig) -> dict[Source, torch.Tensor]:
    return {k: d[k] for k in cfg.inputs}


@torch.no_grad()
def evaluate(model: LightningNet, loader: DataLoader, cfg: DataConfig, device: str, hist: LogitHistogram | None = None) -> dict[str, float]:
    model.eval()
    acc = WindowMetrics(len(cfg.names))
    dt = amp_dtype(device)
    for batch in loader:
        d = decode_batch(batch, cfg, device)
        with torch.autocast(device_type="cuda", dtype=dt or torch.float16, enabled=dt is not None):
            logit, _ = model(inputs_of(d, cfg), d["static"])
        acc.update(logit.float(), d["y"], d["m"], d["pers"])
        if hist is not None:
            hist.add(logit.float(), d["y"], d["m"])
    return acc.result(cfg.names)


def calibrate(best: Path, out: Path, loader: DataLoader, cfg: DataConfig, device: str) -> None:
    ckpt: dict[str, Any] = torch.load(best, map_location="cpu", weights_only=False)
    hist = LogitHistogram(len(cfg.names), device)
    evaluate(load_model(ckpt, device), loader, cfg, device, hist)
    fits = [hist.fit(k) for k in range(len(cfg.names))]
    for k, (name, platt) in enumerate(zip(cfg.names, fits)):
        raw, cal = hist.skill(k), hist.skill(k, platt)
        print(f"{name}: base rate {raw.base_rate:.4f} | raw BSS {raw.bss:+.3f} CSI {raw.best_csi:.3f} | calibrated BSS {cal.bss:+.3f} "
              f"CSI {cal.best_csi:.3f} at {cal.best_threshold:.2f} (a {platt.a:.2f}, b {platt.b:+.2f})", flush=True)
    ckpt["calibration"] = {"a": [p.a for p in fits], "b": [p.b for p in fits], "fitted_on": "validation split"}
    ckpt["product_thresholds"] = {f"w{n}": hist.skill(k, fits[k]).best_threshold for k, n in enumerate(cfg.names)}
    torch.save(ckpt, out)
    print(f"wrote {out}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packed", type=Path, required=True, help="Folder of packed YYYY_MM.zarr stores")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--start", type=str, default=None, help="First anchor (UTC), e.g. 2025-01-01")
    ap.add_argument("--end", type=str, default=None, help="Last anchor (UTC)")
    ap.add_argument("--windows-min", type=int, nargs="+", default=[20, 40, 60, 90, 120], help="Window ends after T; windows start at T+10")
    ap.add_argument("--radii-km", type=float, nargs="+", default=[10, 10, 10, 25, 25], help="One radius per window")
    ap.add_argument("--inputs", nargs="+", type=Source, default=list(DataConfig.inputs), help="Input sources: r120 r300 sat ltg")
    ap.add_argument("--target-kinds", choices=["all", "cg"], default="all", help="Target lightning: in-cloud + ground, or ground only")
    ap.add_argument("--drop", nargs="*", default=["r120=0.1", "sat=0.2", "ltg=0.1"], help="Modality dropout, SOURCE=P")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--crop", type=int, default=192, help="Training crop of the 320 grid (multiple of 2**depth)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--stem-ch", type=int, default=32)
    ap.add_argument("--max-ch", type=int, default=256, help="Channel cap of the deeper levels")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ref-weight", type=float, default=0.1, help="Weight of the auxiliary future-reflectivity loss")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--purge-hours", type=float, default=3.0, help="Gap between training and validation anchors")
    ap.add_argument("--patience", type=int, default=6, help="Epochs without a better validation CSI score")
    ap.add_argument("--max-val", type=int, default=4000, help="Validation anchors per epoch (evenly spaced subset; 0 = all)")
    ap.add_argument("--num-workers", type=int, default=8, help="~0.7 GB of host RAM each, plus 2 prefetched batches")
    ap.add_argument("--val-workers", type=int, default=2)
    ap.add_argument("--run", type=int, default=4, help="Consecutive anchors per run within a batch (1 = plain shuffling)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="Debug: limit training anchors")
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="Debug / speed checks: stop each epoch after N steps")
    ap.add_argument("--no-calibrate", action="store_true", help="Skip the final calibration (speed checks)")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if len(args.radii_km) != len(args.windows_min):
        raise SystemExit("--radii-km needs one radius per window")
    if args.crop and args.crop % 2**args.depth:
        raise SystemExit(f"--crop must be a multiple of {2**args.depth}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cfg = DataConfig(windows_min=tuple(args.windows_min), radii_km=tuple(args.radii_km), inputs=tuple(args.inputs), target_kinds=args.target_kinds)
    stores = PackedStores(args.packed)
    anchors = valid_anchors(stores, cfg, np.datetime64(args.start) if args.start else None, np.datetime64(args.end) if args.end else None)
    train_a, val_a = split_anchors(stores, anchors, args.purge_hours)
    if not (train_a.size and val_a.size):
        raise SystemExit(f"need training and validation anchors (validation: ISO weeks divisible by 5); have {train_a.size} and {val_a.size}")
    if args.max_train:
        train_a = np.sort(np.random.permutation(train_a)[: args.max_train])
    val_eval = val_a[np.linspace(0, val_a.size - 1, args.max_val).astype(int)] if args.max_val and val_a.size > args.max_val else val_a
    print(f"device {args.device}; anchors {anchors.size}: train {train_a.size}, val {val_a.size} (scored per epoch {val_eval.size}); "
          f"outputs {cfg.names}; inputs {[str(s) for s in cfg.inputs]}", flush=True)
    pin = args.device.startswith("cuda")
    train_dl = DataLoader(Samples(stores, train_a, cfg, crop=args.crop or None),
                          batch_sampler=RunBatchSampler(train_a, args.batch_size, args.run, seed=args.seed),
                          num_workers=args.num_workers, pin_memory=pin, persistent_workers=args.num_workers > 0,
                          multiprocessing_context=worker_context(args.num_workers),
                          prefetch_factor=2 if args.num_workers else None)  # ~190 MB per prefetched batch of host RAM
    # not persistent: idle validation workers would hold ~0.7 GB each through training
    val_dl = DataLoader(Samples(stores, val_eval, cfg), batch_size=max(1, args.batch_size // 4), num_workers=args.val_workers, pin_memory=pin,
                        multiprocessing_context=worker_context(args.val_workers))
    drop = {Source(k): float(v) for k, v in (s.split("=") for s in args.drop)}

    model_kwargs = {"branches": {str(k): v for k, v in cfg.channels().items()}, "n_static": 1, "n_windows": len(cfg.names),
                    "n_ref": cfg.n_ref, "base_ch": args.base_ch, "depth": args.depth, "stem_ch": args.stem_ch,
                    "dropout": args.dropout, "max_ch": args.max_ch}
    model = LightningNet(**model_kwargs).to(args.device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.steps_per_epoch or len(train_dl)
    total, warm = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    dt = amp_dtype(args.device)
    scaler = torch.amp.GradScaler("cuda", enabled=dt == torch.float16)
    packed_config = {**asdict(cfg), "inputs": [str(s) for s in cfg.inputs]}
    meta = {"model_kwargs": model_kwargs, "packed_config": packed_config, "arch": "fusion",
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "inputs"},
            "anchors": {"train": int(train_a.size), "val": int(val_a.size), "first": str(stores.slot_time(anchors[0])),
                        "last": str(stores.slot_time(anchors[-1]))}}

    best_score, bad = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, wait, losses, n_seen = time.time(), 0.0, [], 0
        tw = time.time()
        for n, batch in enumerate(tqdm(train_dl, desc=f"epoch {epoch}", mininterval=60, total=steps_per_epoch), 1):
            wait += time.time() - tw
            d = decode_batch(batch, cfg, args.device, drop)
            with torch.autocast(device_type="cuda", dtype=dt or torch.float16, enabled=dt is not None):
                logit, ref = model(inputs_of(d, cfg), d["static"])
                loss = masked_bce(logit, d["y"], d["m"]) + args.ref_weight * ref_loss(ref, d["y_ref"], d["ref_ok"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - args.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            losses.append(float(loss))
            n_seen += logit.shape[0]
            tw = time.time()
            if n >= steps_per_epoch:
                break
        train_sec = time.time() - t0
        metrics = evaluate(ema, val_dl, cfg, args.device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0], "train_sec": train_sec,
                  "samples_per_sec": n_seen / train_sec, "data_wait_frac": wait / train_sec, "val_sec": time.time() - t0 - train_sec,
                  "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if pin else None, **metrics}
        with (args.run_dir / "log.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch {epoch}: {record['samples_per_sec']:.0f} samples/s (data wait {record['data_wait_frac']:.0%}), loss {record['train_loss']:.4f}, "
              f"val bce {metrics['val_bce']:.5f}, score {metrics['score']:.3f} | "
              + " | ".join(f"{n} BSS {metrics.get(f'w{n}_bss', float('nan')):+.3f} CSI {metrics.get(f'w{n}_best_csi', float('nan')):.3f} "
                           f"(pers {metrics.get(f'w{n}_persistence_csi', float('nan')):.3f})" for n in cfg.names), flush=True)
        ckpt = {"model": ema.state_dict(), **meta, "epoch": epoch, "metrics": record,
                "product_thresholds": {f"w{n}": metrics.get(f"w{n}_best_threshold", 0.5) for n in cfg.names}}
        torch.save(ckpt, args.run_dir / "last.pt")
        if metrics["score"] > best_score:
            best_score, bad = metrics["score"], 0
            torch.save(ckpt, args.run_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no better validation CSI score in {args.patience} epochs (best {best_score:.3f})", flush=True)
                break
    if not args.no_calibrate:
        full_val = DataLoader(Samples(stores, val_a, cfg), batch_size=max(1, args.batch_size // 4), num_workers=args.val_workers, pin_memory=pin,
                        multiprocessing_context=worker_context(args.val_workers))
        calibrate(args.run_dir / "best.pt", args.run_dir / "model.pt", full_val, cfg, args.device)


if __name__ == "__main__":
    main()
