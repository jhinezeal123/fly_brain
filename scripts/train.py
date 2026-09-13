from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from fly_depth.data import NYUv2, NYUv2Dataset
from fly_depth.model import DepthHead, FlyDepthModel, MaleCNSVisualModel, depth_metrics


@torch.no_grad()
def evaluate(model: FlyDepthModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for image, depth in loader:
        image, depth = image.to(device), depth.to(device)
        preds.append(torch.exp(model(image)).cpu())
        targets.append(depth.cpu())
    return depth_metrics(torch.cat(preds), torch.cat(targets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(cfg["data"]["dir"])
    store = NYUv2(data_dir / "raw/nyu_depth_v2_labeled.mat", data_dir / "raw/splits.mat")
    output_size = tuple(cfg["model"]["output_size"])
    train_loader = DataLoader(
        NYUv2Dataset(store, "train", output_size),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        NYUv2Dataset(store, "test", output_size),
        batch_size=cfg["train"]["batch_size"],
        num_workers=0,
    )

    brain = MaleCNSVisualModel(
        data_dir / "processed",
        steps=cfg["connectome"]["steps"],
        dt=cfg["connectome"]["dt"],
        tau_init=cfg["connectome"]["tau_init"],
        min_tau=cfg["connectome"]["min_tau"],
        rate_max=cfg["connectome"]["rate_max"],
        input_gain=cfg["connectome"]["input_gain"],
    )
    head = DepthHead(brain.feature_dim, cfg["model"]["hidden"], output_size)
    model = FlyDepthModel(brain, head).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.brain.parameters(), "lr": cfg["train"]["brain_lr"]},
            {"params": model.head.parameters(), "lr": cfg["train"]["head_lr"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total = 0.0
        for image, depth in train_loader:
            image = image.to(device)
            depth = depth.to(device).clamp(0.1, 10.0)
            pred_log_depth = model(image)
            depth_loss = loss_fn(pred_log_depth, torch.log(depth))
            bio_reg = model.brain.regularization(cfg["connectome"]["tau_init"])
            loss = depth_loss + cfg["train"]["biology_reg"] * bio_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            total += float(loss.detach())

        metrics = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch:03d} loss={total / len(train_loader):.4f} "
            f"abs_rel={metrics['abs_rel']:.4f} rmse={metrics['rmse']:.3f} "
            f"delta1={metrics['delta1']:.3f}"
        )

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "fly_depth.pt")
    print("saved checkpoints/fly_depth.pt")


if __name__ == "__main__":
    main()
