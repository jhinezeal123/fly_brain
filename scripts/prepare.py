from __future__ import annotations

import argparse
from pathlib import Path

from fly_depth.connectome import SOURCES, build_visual_connectome, download
from fly_depth.data import NYU_SPLITS_URL, NYU_URL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--eye", choices=["L", "R"], default="R")
    parser.add_argument("--hops", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw = data_dir / "raw"
    processed = data_dir / "processed"

    download(NYU_URL, raw / "nyu_depth_v2_labeled.mat")
    download(NYU_SPLITS_URL, raw / "splits.mat")
    for url in SOURCES.values():
        download(url, raw / Path(url).name)

    target = processed / "visual_connectome.npz"
    if args.rebuild or not target.exists():
        build_visual_connectome(raw, processed, eye=args.eye, max_hops=args.hops)
    else:
        print("processed MaleCNS graph already exists; use --rebuild to replace it")


if __name__ == "__main__":
    main()
