from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fly_depth.connectome import ConnectomeEncoder
from fly_depth.data import NYUv2
from fly_depth.model import DepthHead, depth_metrics


def load_or_build_cache(
    split: str,
    store: NYUv2,
    encoder: ConnectomeEncoder,
    cache_dir: Path,
    output_size: tuple[int, int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = cache_dir / f"{split}_features_steps{encoder.steps}.npz"
    if path.exists():
        cached = np.load(path)
        return cached["features"], cached["depth"]

    indices = store.indices(split)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        images, depths = zip(*(store.get(int(i)) for i in batch_ids))
        features.append(encoder.encode_batch(list(images)))
        targets.extend(store.resize_depth(d, output_size) for d in depths)
        print(f"{split} features {min(start + batch_size, len(indices))}/{len(indices)}")

    x = np.concatenate(features, axis=0).astype(np.float32)
    y = np.stack(targets, axis=0).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=x, depth=y)
    return x, y


@torch.no_grad()
def evaluate(
    model: DepthHead,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for features, depth in loader:
        features = (features.to(device) - mean) / std
        depth = depth.to(device)
        log_depth = model(features)
        preds.append(torch.exp(log_depth).cpu())
        targets.append(depth.cpu())
    return depth_metrics(torch.cat(preds), torch.cat(targets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    data_dir = Path(cfg["data"]["dir"])
    raw = data_dir / "raw"
    processed = data_dir / "processed"
    cache_dir = data_dir / "cache"
    output_size = tuple(cfg["model"]["output_size"])

    encoder = ConnectomeEncoder(processed, steps=cfg["connectome"]["steps"])
    store = NYUv2(raw / "nyu_depth_v2_labeled.mat", raw / "splits.mat")

    train_x, train_y = load_or_build_cache(
        "train", store, encoder, cache_dir, output_size, cfg["connectome"]["batch_size"]
    )
    test_x, test_y = load_or_build_cache(
        "test", store, encoder, cache_dir, output_size, cfg["connectome"]["batch_size"]
    )

    if not np.isfinite(train_x).all() or np.max(np.std(train_x, axis=0)) == 0:
        raise RuntimeError(
            "MaleCNS features are empty/constant; inspect data/processed/meta.json "
            "and increase connectome.steps."
        )

    mean_np = train_x.mean(axis=0, keepdims=True)
    std_np = train_x.std(axis=0, keepdims=True) + 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y)[:, None],
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_x),
        torch.from_numpy(test_y)[:, None],
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])

    model = DepthHead(
        encoder.feature_dim,
        hidden=cfg["model"]["hidden"],
        output_size=output_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total = 0.0
        for features, depth in train_loader:
            features = (features.to(device) - mean) / std
            depth = depth.to(device).clamp(0.1, 10.0)
            pred_log_depth = model(features)
            loss = loss_fn(pred_log_depth, torch.log(depth))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)

        metrics = evaluate(model, test_loader, mean, std, device)
        print(
            f"epoch={epoch:03d} loss={total / len(train_loader):.4f} "
            f"abs_rel={metrics['abs_rel']:.4f} rmse={metrics['rmse']:.3f} "
            f"delta1={metrics['delta1']:.3f}"
        )

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": mean_np,
            "feature_std": std_np,
            "feature_dim": encoder.feature_dim,
            "output_size": output_size,
            "hidden": cfg["model"]["hidden"],
            "steps": cfg["connectome"]["steps"],
        },
        ckpt_dir / "depth_head.pt",
    )
    print("saved checkpoints/depth_head.pt")


if __name__ == "__main__":
    main()
