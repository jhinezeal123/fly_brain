# fly_brain

Minimal sandbox for testing depth estimation from a single RGB image with a replaceable fly-inspired visual encoder.

## Quick start

```bash
git clone https://github.com/jhinezeal123/fly_brain.git
cd fly_brain
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python scripts/train.py --config configs/base.yaml
```

## Pipeline

```text
RGB image
   ↓
retina image encoder
   ↓
fly visual representation
   ↓
depth decoder
   ↓
relative depth map
```

`FlyVisualEncoder` is currently a lightweight placeholder. The next step is to replace it with MaleCNS/connectome-derived visual activity without changing the rest of the pipeline.

## Structure

```text
configs/base.yaml        experiment config
src/fly_depth/model.py   image → fly representation → depth
src/fly_depth/data.py    tiny synthetic monocular-depth dataset
scripts/train.py         training entry point
```
