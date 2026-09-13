from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

NYU_URL = "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_SPLITS_URL = "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"


class NYUv2:
    """Official NYU Depth V2 labeled RGB-D data and official 795/654 split."""

    def __init__(self, mat_path: Path, splits_path: Path) -> None:
        self.mat_path = Path(mat_path)
        self.splits = scipy.io.loadmat(splits_path)
        self._h5: h5py.File | None = None

    def indices(self, split: str) -> np.ndarray:
        key = {"train": "trainNdxs", "test": "testNdxs"}[split]
        return np.asarray(self.splits[key]).reshape(-1).astype(np.int64) - 1

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.mat_path, "r")
        return self._h5

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        f = self._file()
        image = np.asarray(f["images"][index])
        depth = np.asarray(f["depths"][index], dtype=np.float32)
        if image.shape[0] == 3:
            image = np.transpose(image, (2, 1, 0))
        if depth.shape[0] != image.shape[0]:
            depth = depth.T
        return image.astype(np.uint8), depth.astype(np.float32)


class NYUv2Dataset(Dataset):
    def __init__(self, store: NYUv2, split: str, output_size: tuple[int, int]) -> None:
        self.store = store
        self.ids = store.indices(split)
        self.output_size = tuple(output_size)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, depth = self.store.get(int(self.ids[item]))
        image_t = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth.copy())[None, None]
        depth_t = F.interpolate(depth_t, size=self.output_size, mode="bilinear", align_corners=False)[0]
        return image_t, depth_t
