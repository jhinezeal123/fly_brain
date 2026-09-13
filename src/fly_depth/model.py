from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RetinaImageEncoder(nn.Module):
    """Encode one RGB image into compact retina-like visual features."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


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
    """RGB image -> fly-style visual activity -> relative depth in [0, 1]."""

    def __init__(self, channels: int = 32, activity_dim: int = 64) -> None:
        super().__init__()
        self.retina = RetinaImageEncoder(channels)
        self.fly = FlyVisualEncoder(channels, activity_dim)
        self.decoder = DepthDecoder(activity_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        retina_features = self.retina(image)
        activity = self.fly(retina_features)
        return self.decoder(activity, image.shape[-2:])
