from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RetinaMotionEncoder(nn.Module):
    """Encode two RGB frames into a compact motion-aware representation."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, channels, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, frame0: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
        x = torch.cat([frame0, frame1, frame1 - frame0], dim=1)
        return self.net(x)


class FlyVisualEncoder(nn.Module):
    """Placeholder for a future MaleCNS/connectome-derived visual encoder.

    Keep this interface stable: input is retina features, output is neural activity.
    """

    def __init__(self, channels: int = 32, activity_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, activity_dim, 1),
            nn.GELU(),
            nn.Conv2d(activity_dim, activity_dim, 3, padding=1, groups=activity_dim),
            nn.GELU(),
        )

    def forward(self, retina_features: torch.Tensor) -> torch.Tensor:
        return self.net(retina_features)


class DepthDecoder(nn.Module):
    def __init__(self, activity_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(activity_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, activity: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        depth = self.net(activity)
        depth = F.interpolate(depth, size=output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(depth)


class FlyDepthModel(nn.Module):
    """frame[t-1], frame[t] -> fly-style activity -> relative depth in [0, 1]."""

    def __init__(self, channels: int = 32, activity_dim: int = 64) -> None:
        super().__init__()
        self.retina = RetinaMotionEncoder(channels)
        self.fly = FlyVisualEncoder(channels, activity_dim)
        self.decoder = DepthDecoder(activity_dim)

    def forward(self, frame0: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
        retina_features = self.retina(frame0, frame1)
        activity = self.fly(retina_features)
        return self.decoder(activity, frame0.shape[-2:])
