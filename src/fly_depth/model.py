from __future__ import annotations

import torch
from torch import nn


class DepthHead(nn.Module):
    """The only trainable part of the experiment."""

    def __init__(self, feature_dim: int, hidden: int, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_size[0] * output_size[1]),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        log_depth = self.net(features)
        return log_depth.view(-1, 1, *self.output_size)


def depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.clamp(0.1, 10.0)
    target = target.clamp(0.1, 10.0)
    abs_rel = ((pred - target).abs() / target).mean()
    rmse = torch.sqrt(((pred - target) ** 2).mean())
    ratio = torch.maximum(pred / target, target / pred)
    delta1 = (ratio < 1.25).float().mean()
    return {
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "delta1": float(delta1),
    }
