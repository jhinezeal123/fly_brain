from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from fly_depth.data import SyntheticMotionDepthDataset
from fly_depth.model import FlyDepthModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticMotionDepthDataset(
        length=cfg["data"]["samples"],
        size=cfg["data"]["image_size"],
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    model = FlyDepthModel(
        channels=cfg["model"]["retina_channels"],
        activity_dim=cfg["model"]["activity_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = nn.L1Loss()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for frame0, frame1, depth in loader:
            frame0, frame1, depth = frame0.to(device), frame1.to(device), depth.to(device)
            pred = model(frame0, frame1)
            loss = loss_fn(pred, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch={epoch} loss={total_loss / len(loader):.4f} device={device}")


if __name__ == "__main__":
    main()
