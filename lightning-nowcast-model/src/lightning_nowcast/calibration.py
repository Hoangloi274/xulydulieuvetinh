"""Per-window Platt calibration, p = sigmoid(a * logit + b), fitted on binned validation logits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

LOGIT_RANGE = 16.0
N_BINS = 640  # 0.05-wide logit bins; logits beyond +-16 fall into the end bins
THRESHOLDS = np.round(np.arange(0.01, 0.951, 0.01), 2)


@dataclass(frozen=True)
class Platt:
    a: float
    b: float

    def apply(self, logit: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * logit + self.b)))


@dataclass(frozen=True)
class Skill:
    base_rate: float
    bss: float
    best_threshold: float
    best_csi: float


class LogitHistogram:
    """Counts of event / no-event pixels per logit bin and window, accumulated over batches."""

    def __init__(self, n_windows: int, device: str | torch.device) -> None:
        self.pos = torch.zeros(n_windows, N_BINS, dtype=torch.float64, device=device)
        self.neg = torch.zeros_like(self.pos)

    @torch.no_grad()
    def add(self, logit: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> None:
        idx = ((logit.float() + LOGIT_RANGE) * (N_BINS / (2 * LOGIT_RANGE))).long().clamp(0, N_BINS - 1)
        for k in range(logit.shape[1]):
            mk, yk = m[:, k] > 0.5, y[:, k] > 0.5
            self.pos[k] += torch.bincount(idx[:, k][mk & yk], minlength=N_BINS).double()
            self.neg[k] += torch.bincount(idx[:, k][mk & ~yk], minlength=N_BINS).double()

    def fit(self, k: int) -> Platt:
        a, b = fit_platt(bin_centres(), self.pos[k].cpu().numpy(), self.neg[k].cpu().numpy())
        return Platt(a, b)

    def skill(self, k: int, platt: Platt | None = None) -> Skill:
        x = bin_centres()
        prob = platt.apply(x) if platt else 1.0 / (1.0 + np.exp(-x))
        return skill(prob, self.pos[k].cpu().numpy(), self.neg[k].cpu().numpy())


def bin_centres() -> np.ndarray:
    return -LOGIT_RANGE + (np.arange(N_BINS) + 0.5) * (2 * LOGIT_RANGE / N_BINS)


def fit_platt(x: np.ndarray, pos: np.ndarray, neg: np.ndarray, iters: int = 100) -> tuple[float, float]:
    """Weighted logistic regression of the label on one feature (damped Newton on binned counts)."""
    keep = (pos + neg) > 0
    x, pos, neg = x[keep], pos[keep], neg[keep]
    feats = np.stack([x, np.ones_like(x)], axis=1)

    def nll(theta: np.ndarray) -> float:
        z = feats @ theta
        return float((pos * np.logaddexp(0.0, -z) + neg * np.logaddexp(0.0, z)).sum())

    theta = np.array([1.0, 0.0])
    loss = nll(theta)
    for _ in range(iters):
        p = 0.5 * (1.0 + np.tanh(0.5 * (feats @ theta)))
        grad = feats.T @ (pos - (pos + neg) * p)
        hess = (feats * ((pos + neg) * p * (1 - p))[:, None]).T @ feats + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        scale = 1.0
        while scale > 1e-6 and nll(theta + scale * step) > loss:
            scale *= 0.5
        theta = theta + scale * step
        new_loss = nll(theta)
        if loss - new_loss < 1e-10 * max(loss, 1.0):
            break
        loss = new_loss
    return float(theta[0]), float(theta[1])


def skill(prob: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> Skill:
    """Brier skill and best pooled CSI of per-bin probabilities."""
    n = pos.sum() + neg.sum()
    base = pos.sum() / n
    brier = float(((prob**2) * neg + ((1 - prob) ** 2) * pos).sum() / n)
    csi = [(pos[prob >= t].sum()) / max(pos[prob >= t].sum() + neg[prob >= t].sum() + pos[prob < t].sum(), 1.0) for t in THRESHOLDS]
    best = int(np.argmax(csi))
    return Skill(float(base), float(1.0 - brier / (base * (1 - base))), float(THRESHOLDS[best]), float(csi[best]))
