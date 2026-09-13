# fly_brain

Minimal experiment for **single-image depth estimation using the real MaleCNS v1.0 connectome**.

## What is real

- **Brain wiring:** official MaleCNS v1.0 neuron annotations and synapse-count graph.
- **Visual input nodes:** real R1-R6 photoreceptors, mapped through their published connections to L1 optic columns.
- **Training data:** official NYU Depth V2 RGB + Kinect depth, using the official 795 train / 654 test split.
- The connectome is **frozen**. Optimizer updates only the small depth head.

MaleCNS is anatomy, not recorded neural activity. This repo therefore uses the connectome as a fixed graph transform; it does **not** claim to simulate a living fly brain. The only engineered sensory assumption is mapping an RGB camera image onto one compound-eye column map.

## Pipeline

```text
NYU RGB image
    ↓
right-eye optic columns
    ↓
real R1-R6 MaleCNS photoreceptors
    ↓
fixed MaleCNS visual synapse graph
    ↓
visual_projection neurons
    ↓
trainable depth head
    ↓
depth in meters
```

## Run

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python scripts/prepare.py
python scripts/train.py
python scripts/infer.py path/to/image.jpg
```

`prepare.py` downloads about 4 GB of source data and builds the visual sparse graph. Raw files are not committed.

Outputs:

```text
checkpoints/depth_head.pt
outputs/depth_mm.png
outputs/depth_mm.npy
```

## GPU

GPU is optional. MaleCNS feature extraction uses CPU sparse matrices. A GPU only speeds up training the small depth head. For graph preparation, **16 GB RAM is recommended**.

## Sources

- MaleCNS v1.0: https://male-cns.janelia.org/download/
- MaleCNS optic-column assignments: https://github.com/flyconnectome/2025malecns
- NYU Depth V2: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
