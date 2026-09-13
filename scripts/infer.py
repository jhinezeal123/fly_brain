from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fly_depth.model import DepthHead, FlyDepthModel, MaleCNSVisualModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="checkpoints/fly_depth.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="outputs/depth_mm.png")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    brain = MaleCNSVisualModel(
        Path(args.data_dir) / "processed",
        steps=cfg["connectome"]["steps"],
        dt=cfg["connectome"]["dt"],
        tau_init=cfg["connectome"]["tau_init"],
        min_tau=cfg["connectome"]["min_tau"],
        rate_max=cfg["connectome"]["rate_max"],
        input_gain=cfg["connectome"]["input_gain"],
    )
    head = DepthHead(brain.feature_dim, cfg["model"]["hidden"], tuple(cfg["model"]["output_size"]))
    model = FlyDepthModel(brain, head)
    model.load_state_dict(ckpt["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().to(device) / 255.0
    with torch.no_grad():
        depth = torch.exp(model(x))
        depth = F.interpolate(depth, size=image.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), depth.astype(np.float32))
    Image.fromarray(np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)).save(out)
    print(f"saved {out} and {out.with_suffix('.npy')}")


if __name__ == "__main__":
    main()
