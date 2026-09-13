from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


class SyntheticImageDepthDataset(Dataset):
    """Tiny monocular dataset with visible foreground/background depth cues."""

    def __init__(self, length: int = 512, size: int = 64) -> None:
        self.length = length
        self.size = size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        _ = index
        h = w = self.size

        # Far background: dim, noisy texture.
        image = 0.15 + 0.15 * torch.rand(3, h, w)
        depth = torch.ones(1, h, w)

        # Nearer objects are larger and brighter, giving the single image
        # simple monocular cues that a tiny model can learn.
        near_depth = random.uniform(0.2, 0.7)
        scale = 1.0 - near_depth
        min_side = max(4, int(self.size * (0.15 + 0.25 * scale)))
        max_side = max(min_side + 1, int(self.size * (0.25 + 0.40 * scale)))

        box_h = random.randint(min_side, min(max_side, h - 1))
        box_w = random.randint(min_side, min(max_side, w - 1))
        y0 = random.randint(0, h - box_h)
        x0 = random.randint(0, w - box_w)

        color = torch.rand(3, 1, 1) * 0.35 + (0.55 + 0.25 * scale)
        patch = color.expand(3, box_h, box_w).clone()
        patch += 0.05 * torch.rand_like(patch)
        image[:, y0 : y0 + box_h, x0 : x0 + box_w] = patch.clamp(0.0, 1.0)
        depth[:, y0 : y0 + box_h, x0 : x0 + box_w] = near_depth

        return image, depth
