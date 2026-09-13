# fly_brain

Monocular depth experiment constrained by the real **MaleCNS v1.0 visual connectome**.

```text
NYU Depth V2 RGB
      ↓
real R1-R6 photoreceptors
      ↓
MaleCNS visual topology + synapse counts   (fixed)
      ↓
cell-type dynamics                         (trainable)
      ↓
visual projection neurons
      ↓
small depth head                           (trainable)
      ↓
depth in meters
```

## What is trained?

The connectome is **not rewired**. Neuron IDs, directed edges and synapse counts stay fixed.
Training only fits:

- `tau[cell_type]`
- `bias[cell_type]`
- `gain[src_type, dst_type]`
- the small depth head

This follows the connectome-constrained idea used by FlyVis rather than treating the wiring diagram as a complete biological simulator.

## Run

```bash
git clone https://github.com/jhinezeal123/fly_brain.git
cd fly_brain
pip install -e .

python scripts/prepare.py
python scripts/train.py
python scripts/infer.py path/to/image.jpg
```

`prepare.py` downloads official MaleCNS tables and the official NYU Depth V2 labeled RGB-D dataset, then extracts the retina-rooted visual subgraph.

A CUDA GPU is strongly recommended for training. `batch_size: 1` is intentional because sparse connectome backprop is memory-heavy.

## Important boundary

MaleCNS provides anatomy/connectivity, not measured membrane dynamics. `tau`, biases, transmitter signs and type-pair gains are explicit computational assumptions or fitted parameters. RGB-camera luminance → compound-eye sampling is also an engineered adapter. The repo tests whether a **connectome-constrained visual network** can learn a depth representation; it does not claim to reproduce a living fly brain.
