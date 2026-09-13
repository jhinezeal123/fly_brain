from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fly_depth.connectome import ConnectomeEncoder
from fly_depth.model import DepthHead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="checkpoints/depth_head.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="outputs/depth_mm.png")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder = ConnectomeEncoder(
        Path(args.data_dir) / "processed",
        steps=int(ckpt["steps"]),
    )
    features = encoder.encode_batch([image])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthHead(
        int(ckpt["feature_dim"]),
        int(ckpt["hidden"]),
        tuple(ckpt["output_size"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean = torch.from_numpy(ckpt["feature_mean"]).to(device)
    std = torch.from_numpy(ckpt["feature_std"]).to(device)
    x = torch.from_numpy(features).to(device)
    with torch.no_grad():
        depth = torch.exp(model((x - mean) / std))
        depth = F.interpolate(
            depth, size=image.shape[:2], mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), depth.astype(np.float32))
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(out)
    print(f"saved {out} and {out.with_suffix('.npy')}")


if __name__ == "__main__":
    main()
