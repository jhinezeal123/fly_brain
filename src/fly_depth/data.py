from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


class SyntheticMotionDepthDataset(Dataset):
    """Tiny synthetic dataset: closer rectangles move more between two frames."""

    def __init__(self, length: int = 512, size: int = 64) -> None:
        self.length = length
        self.size = size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        _ = index
        h = w = self.size
        texture = torch.rand(3, h, w)

        depth = torch.ones(1, h, w)
        mask = torch.zeros(1, h, w)

        box_h = random.randint(h // 5, h // 2)
        box_w = random.randint(w // 5, w // 2)
        y0 = random.randint(0, h - box_h)
        x0 = random.randint(0, w - box_w)
        mask[:, y0 : y0 + box_h, x0 : x0 + box_w] = 1.0

        near_depth = random.uniform(0.2, 0.6)
        depth = depth * (1.0 - mask) + near_depth * mask

        far_shift = random.choice([-1, 1])
        near_shift = far_shift * random.randint(3, 6)

        frame0 = texture
        background = torch.roll(texture, shifts=far_shift, dims=2)
        foreground = torch.roll(texture * mask, shifts=near_shift, dims=2)
        shifted_mask = torch.roll(mask, shifts=near_shift, dims=2)
        frame1 = background * (1.0 - shifted_mask) + foreground

        return frame0, frame1, depth
