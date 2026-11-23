"""The lightning nowcaster: one convolutional stem per input source, fused into a residual U-Net with one output per
forecast window (and an auxiliary future-reflectivity head used in training only).

- Each source's frames are stacked along channels (all times seen at once, so motion and growth are learned from
  frame differences, as in RainNet and LightningCast) and get their own two-layer stem, so a missing source can be
  zeroed on its own.
- Each output channel is a time window predicted directly: no autoregressive rollout.
- Five 2x downsamplings with channels capped at max_ch: context over the whole 640 km domain at ~10 M parameters.
  Larger models, global attention and time conditioning were tried and did not score better
  (archive/development).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .grid import Source


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.SiLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _gn(cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.body(x) + self.skip(x))


class LightningNet(nn.Module):
    def __init__(self, branches: dict[str, int], n_static: int, n_windows: int, n_ref: int = 6,
                 base_ch: int = 32, depth: int = 5, stem_ch: int = 32, dropout: float = 0.1, max_ch: int = 256) -> None:
        super().__init__()
        self.names = [str(k) for k in branches]
        self.stems = nn.ModuleDict({str(k): nn.Sequential(
            nn.Conv2d(c, stem_ch, 3, padding=1), _gn(stem_ch), nn.SiLU(),
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1), _gn(stem_ch), nn.SiLU()) for k, c in branches.items()})
        chs = [min(base_ch * 2**i, max_ch) for i in range(depth + 1)]
        self.enc = nn.ModuleList([ResBlock(len(branches) * stem_ch + n_static, chs[0])] + [ResBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.mid = ResBlock(chs[-1], chs[-1], dropout=dropout)
        self.dec = nn.ModuleList([ResBlock(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], n_windows, 1)
        self.head_ref = nn.Conv2d(chs[0], n_ref, 1)
        self.depth = depth
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, -4.6)  # start from the climatological rate (~1%), not 50%

    def forward(self, inputs: dict[Source, torch.Tensor], static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs: source -> (B, C, H, W); static (B, S, H, W); H, W divisible by 2**depth.
        Returns window logits (B, n_windows, H, W) and auxiliary reflectivity (B, n_ref, H, W)."""
        x = torch.cat([self.stems[k](inputs[Source(k)]) for k in self.names] + [static], dim=1)
        skips = []
        for i, block in enumerate(self.enc):
            if i > 0:
                x = F.avg_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        x = self.mid(x)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x), self.head_ref(x)


def load_model(ckpt: dict[str, Any], device: str | torch.device = "cpu") -> LightningNet:
    """The network of a checkpoint, with its weights, in eval mode."""
    if ckpt.get("arch", "fusion") != "fusion":
        raise ValueError(f"checkpoint architecture {ckpt['arch']!r} is not in this package (see archive/development)")
    model = LightningNet(**ckpt["model_kwargs"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model
