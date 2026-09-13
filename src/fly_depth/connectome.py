from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import scipy.sparse as sp

MALECNS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SOURCES = {
    "annotations": MALECNS_BASE + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "weights": MALECNS_BASE + "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "optic_columns": "https://raw.githubusercontent.com/flyconnectome/2025malecns/main/supplemental_data/optic-column-type-assignments-v1.0.xlsx",
}
VISUAL_SUPERCLASSES = {"ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal"}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {path.name}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def build_visual_connectome(raw_dir: Path, out_dir: Path, eye: str = "R") -> None:
    """Build a fixed visual graph from official MaleCNS v1.0 tables.

    The graph is not a neural-dynamics simulator. Each row is the published
    synapse-count distribution from one neuron to its visual-graph targets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = raw_dir / Path(SOURCES["annotations"]).name
    weights_path = raw_dir / Path(SOURCES["weights"]).name
    optic_path = raw_dir / Path(SOURCES["optic_columns"]).name

    ann = pd.read_feather(annotations_path)
    required = {"bodyId", "status", "superclass", "type"}
    missing = required - set(ann.columns)
    if missing:
        raise RuntimeError(f"MaleCNS annotation columns missing: {sorted(missing)}")

    visual = ann[
        (ann["status"] == "Traced")
        & ann["superclass"].isin(VISUAL_SUPERCLASSES)
    ].copy()
    visual = visual.sort_values("bodyId").reset_index(drop=True)
    node_ids = visual["bodyId"].to_numpy(dtype=np.int64)
    value_set = pa.array(node_ids)

    pre_chunks: list[np.ndarray] = []
    post_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    full_out = np.zeros(len(node_ids), dtype=np.float64)

    reader = ipc.open_file(pa.memory_map(str(weights_path)))
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        pre = pc.fill_null(pc.index_in(batch["body_pre"], value_set=value_set), -1).to_numpy()
        post = pc.fill_null(pc.index_in(batch["body_post"], value_set=value_set), -1).to_numpy()
        batch_weight = np.asarray(batch["weight"].to_numpy(), dtype=np.float32)

        pre_valid = pre >= 0
        np.add.at(full_out, pre[pre_valid], batch_weight[pre_valid])

        mask = pre_valid & (post >= 0)
        if mask.any():
            pre_chunks.append(np.asarray(pre[mask], dtype=np.int32))
            post_chunks.append(np.asarray(post[mask], dtype=np.int32))
            weight_chunks.append(batch_weight[mask])
        if (i + 1) % 20 == 0 or i + 1 == reader.num_record_batches:
            print(f"connectome batches {i + 1}/{reader.num_record_batches}")

    pre = np.concatenate(pre_chunks)
    post = np.concatenate(post_chunks)
    weights = np.concatenate(weight_chunks)
    raw_graph = sp.coo_matrix(
        (weights, (pre, post)),
        shape=(len(node_ids), len(node_ids)),
        dtype=np.float32,
    ).tocsr()
    raw_graph.sum_duplicates()

    inv = np.zeros(len(node_ids), dtype=np.float32)
    valid = full_out > 0
    inv[valid] = (1.0 / full_out[valid]).astype(np.float32)
    graph = sp.diags(inv) @ raw_graph
    graph = graph.tocsr()

    output_indices = np.flatnonzero(
        visual["superclass"].to_numpy() == "visual_projection"
    ).astype(np.int32)

    optic = pd.read_excel(optic_path)
    pattern = re.compile(rf"^ME_{eye}_col_(\d+)_(\d+)$")
    records: list[tuple[int, int, int]] = []
    for row in optic.itertuples(index=False):
        name = str(getattr(row, "column"))
        match = pattern.match(name)
        if not match:
            continue
        l1 = getattr(row, "L1")
        if pd.isna(l1) or int(l1) < 0:
            continue
        records.append((int(match.group(1)), int(match.group(2)), int(l1)))

    id_to_idx = {int(body): i for i, body in enumerate(node_ids)}
    r1_mask = (
        (visual["superclass"].to_numpy() == "ol_sensory")
        & (visual["type"].fillna("").to_numpy() == "R1-R6")
    )

    csc = raw_graph.tocsc()
    seed_rows: list[int] = []
    seed_cols: list[int] = []
    seed_vals: list[float] = []
    coords: list[tuple[int, int]] = []

    for y, x, l1_body in records:
        l1_idx = id_to_idx.get(l1_body)
        if l1_idx is None:
            continue
        start, end = csc.indptr[l1_idx], csc.indptr[l1_idx + 1]
        incoming = csc.indices[start:end]
        r1 = incoming[r1_mask[incoming]]
        if len(r1) == 0:
            continue
        column_idx = len(coords)
        coords.append((y, x))
        value = 1.0 / len(r1)
        seed_rows.extend([column_idx] * len(r1))
        seed_cols.extend(r1.tolist())
        seed_vals.extend([value] * len(r1))

    if not coords:
        raise RuntimeError("No optic columns could be mapped to R1-R6 photoreceptors")

    seed_matrix = sp.csr_matrix(
        (np.asarray(seed_vals, dtype=np.float32), (seed_rows, seed_cols)),
        shape=(len(coords), len(node_ids)),
        dtype=np.float32,
    )

    sp.save_npz(out_dir / "visual_graph.npz", graph)
    sp.save_npz(out_dir / "seed_matrix.npz", seed_matrix)
    np.savez(
        out_dir / "meta.npz",
        node_ids=node_ids,
        output_indices=output_indices,
        column_coords=np.asarray(coords, dtype=np.int32),
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset": "MaleCNS v1.0",
                "eye": eye,
                "visual_neurons": int(len(node_ids)),
                "visual_edges": int(raw_graph.nnz),
                "optic_columns": int(len(coords)),
                "visual_projection_outputs": int(len(output_indices)),
                "visual_superclasses": sorted(VISUAL_SUPERCLASSES),
                "input": "actual R1-R6 photoreceptors inferred from their published synapses onto L1 optic-column neurons",
                "propagation": "fixed synapse-count graph normalized by each neuron's full released outgoing synapse count; no trainable connectome weights",
                "sources": SOURCES,
            },
            indent=2,
        )
        + "\n"
    )
    print((out_dir / "meta.json").read_text())


class ConnectomeEncoder:
    """Deterministic image -> MaleCNS visual-projection feature transform."""

    def __init__(self, processed_dir: Path, steps: int = 2) -> None:
        self.graph = sp.load_npz(processed_dir / "visual_graph.npz").tocsr()
        self.seed = sp.load_npz(processed_dir / "seed_matrix.npz").tocsr()
        meta = np.load(processed_dir / "meta.npz")
        self.output_indices = meta["output_indices"].astype(np.int64)
        self.coords = meta["column_coords"].astype(np.float32)
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError("steps must be >= 1")

    @property
    def feature_dim(self) -> int:
        return int(len(self.output_indices))

    def _sample_columns(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape HxWx3")
        if image.max() > 1.5:
            image = image / 255.0

        # RGB camera luminance is only an input proxy. MaleCNS provides the wiring,
        # not a calibrated camera-to-photoreceptor transfer function.
        lum = image.mean(axis=2)
        h, w = lum.shape
        ys = self.coords[:, 0]
        xs = self.coords[:, 1]
        ys = (ys - ys.min()) / max(float(ys.max() - ys.min()), 1.0) * (h - 1)
        xs = (xs - xs.min()) / max(float(xs.max() - xs.min()), 1.0) * (w - 1)

        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)
        wy = ys - y0
        wx = xs - x0

        return (
            lum[y0, x0] * (1 - wy) * (1 - wx)
            + lum[y1, x0] * wy * (1 - wx)
            + lum[y0, x1] * (1 - wy) * wx
            + lum[y1, x1] * wy * wx
        ).astype(np.float32)

    def encode_batch(self, images: list[np.ndarray] | np.ndarray) -> np.ndarray:
        columns = np.stack([self._sample_columns(img) for img in images], axis=0)
        state = self.seed.T.dot(columns.T).T
        for _ in range(self.steps):
            state = self.graph.T.dot(state.T).T
        return np.asarray(state[:, self.output_indices], dtype=np.float32)
