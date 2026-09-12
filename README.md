# fly_brain

Minimal sandbox for testing whether fly-inspired visual activity can encode relative depth from motion.

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
frame[t-1], frame[t]
        ↓
retina / motion encoder
        ↓
fly visual representation
        ↓
tiny depth decoder
        ↓
relative depth map
```

`FlyVisualEncoder` is currently a lightweight placeholder so the pipeline runs end-to-end. Replace that module with MaleCNS/connectome-derived neural activity later without changing the training pipeline.

## Structure

```text
configs/base.yaml        experiment config
src/fly_depth/model.py   retina → fly representation → depth
src/fly_depth/data.py    tiny synthetic motion-depth dataset
scripts/train.py         training entry point
```

Main experiment: compare the full fly visual representation against linear probes and neuron/pathway ablations.
