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

MALECNS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SOURCES = {
    "annotations": MALECNS_BASE + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": MALECNS_BASE + "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": MALECNS_BASE + "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "optic_columns": "https://raw.githubusercontent.com/flyconnectome/2025malecns/main/supplemental_data/optic-column-type-assignments-v1.0.xlsx",
}
VISUAL_SUPERCLASSES = {"ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal"}
NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {path.name}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def _optic_records(optic: pd.DataFrame, eye: str) -> list[tuple[int, int, int]]:
    pattern = re.compile(rf"^ME_{eye}_col_(\d+)_(\d+)$")
    records: list[tuple[int, int, int]] = []
    for row in optic.itertuples(index=False):
        match = pattern.match(str(getattr(row, "column")))
        if not match:
            continue
        l1 = getattr(row, "L1")
        if pd.isna(l1) or int(l1) < 0:
            continue
        records.append((int(match.group(1)), int(match.group(2)), int(l1)))
    return records


def build_visual_connectome(
    raw_dir: Path,
    out_dir: Path,
    eye: str = "R",
    max_hops: int = 8,
) -> None:
    """Compile a retina-rooted MaleCNS visual subgraph for differentiable training.

    Fixed biological constraints: neuron identities, directed topology and published
    synapse counts. Trainable dynamics are added later by ``MaleCNSVisualModel``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = pd.read_feather(raw_dir / Path(SOURCES["annotations"]).name)
    nt = pd.read_feather(raw_dir / Path(SOURCES["neurotransmitters"]).name)
    optic = pd.read_excel(raw_dir / Path(SOURCES["optic_columns"]).name)
    weights_path = raw_dir / Path(SOURCES["weights"]).name

    required = {"bodyId", "status", "superclass", "type"}
    missing = required - set(ann.columns)
    if missing:
        raise RuntimeError(f"MaleCNS annotation columns missing: {sorted(missing)}")

    visual = ann[(ann["status"] == "Traced") & ann["superclass"].isin(VISUAL_SUPERCLASSES)].copy()
    visual = visual.sort_values("bodyId").reset_index(drop=True)
    node_ids_all = visual["bodyId"].to_numpy(dtype=np.int64)
    value_set = pa.array(node_ids_all)

    pre_chunks: list[np.ndarray] = []
    post_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []
    reader = ipc.open_file(pa.memory_map(str(weights_path)))
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        pre = pc.fill_null(pc.index_in(batch["body_pre"], value_set=value_set), -1).to_numpy()
        post = pc.fill_null(pc.index_in(batch["body_post"], value_set=value_set), -1).to_numpy()
        count = np.asarray(batch["weight"].to_numpy(), dtype=np.float32)
        mask = (pre >= 0) & (post >= 0)
        if mask.any():
            pre_chunks.append(np.asarray(pre[mask], dtype=np.int32))
            post_chunks.append(np.asarray(post[mask], dtype=np.int32))
            count_chunks.append(count[mask])
        if (i + 1) % 20 == 0 or i + 1 == reader.num_record_batches:
            print(f"connectome batches {i + 1}/{reader.num_record_batches}")

    pre_all = np.concatenate(pre_chunks)
    post_all = np.concatenate(post_chunks)
    count_all = np.concatenate(count_chunks)

    # Map actual R1-R6 bodies to optic columns through their published input to L1.
    optic_records = _optic_records(optic, eye)
    l1_bodies = {record[2] for record in optic_records}
    id_to_all = {int(body): i for i, body in enumerate(node_ids_all)}
    r1_mask = (
        (visual["superclass"].to_numpy() == "ol_sensory")
        & (visual["type"].fillna("").to_numpy() == "R1-R6")
    )
    incoming: dict[int, list[int]] = {}
    for edge_idx, post_idx in enumerate(post_all):
        if int(node_ids_all[post_idx]) in l1_bodies:
            incoming.setdefault(int(post_idx), []).append(edge_idx)

    column_rows: list[tuple[int, int, list[int]]] = []
    seed_nodes: set[int] = set()
    for y, x, l1_body in optic_records:
        l1_idx = id_to_all.get(l1_body)
        if l1_idx is None:
            continue
        edges = incoming.get(l1_idx, [])
        r1 = [int(pre_all[e]) for e in edges if r1_mask[int(pre_all[e])]]
        if not r1:
            continue
        column_rows.append((y, x, r1))
        seed_nodes.update(r1)
    if not column_rows:
        raise RuntimeError("No optic columns could be mapped to MaleCNS R1-R6 photoreceptors")

    # Keep the real directed circuit reachable from the stimulated eye.
    outgoing: list[list[int]] = [[] for _ in range(len(node_ids_all))]
    for edge_idx, pre_idx in enumerate(pre_all):
        outgoing[int(pre_idx)].append(edge_idx)
    reachable = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(max_hops):
        nxt: set[int] = set()
        for node in frontier:
            for edge_idx in outgoing[node]:
                nxt.add(int(post_all[edge_idx]))
        nxt -= reachable
        reachable |= nxt
        frontier = nxt
        if not frontier:
            break

    keep_old = np.asarray(sorted(reachable), dtype=np.int32)
    old_to_new = np.full(len(node_ids_all), -1, dtype=np.int32)
    old_to_new[keep_old] = np.arange(len(keep_old), dtype=np.int32)
    node_ids = node_ids_all[keep_old]
    kept_visual = visual.iloc[keep_old].reset_index(drop=True)

    edge_mask = (old_to_new[pre_all] >= 0) & (old_to_new[post_all] >= 0)
    edge_pre = old_to_new[pre_all[edge_mask]].astype(np.int32)
    edge_post = old_to_new[post_all[edge_mask]].astype(np.int32)
    edge_count = count_all[edge_mask].astype(np.float32)

    # Published counts stay fixed; postsynaptic normalization is an explicit numerical
    # dynamics assumption so raw contact counts do not make the rate model explode.
    total_in = np.zeros(len(node_ids), dtype=np.float32)
    np.add.at(total_in, edge_post, edge_count)
    edge_weight = edge_count / np.maximum(total_in[edge_post], 1.0)

    nt_by_body = nt.set_index("body") if "body" in nt.columns else pd.DataFrame()
    signs = np.zeros(len(node_ids), dtype=np.float32)
    for i, body in enumerate(node_ids):
        value = None
        if not nt_by_body.empty and int(body) in nt_by_body.index:
            value = nt_by_body.loc[int(body)].get("consensus_nt")
        name = "unknown" if value is None or pd.isna(value) else str(value).lower()
        signs[i] = NT_SIGN.get(name, 0.0)
    edge_sign = signs[edge_pre].astype(np.float32)

    type_names_raw = []
    for row in kept_visual.itertuples(index=False):
        cell_type = getattr(row, "type")
        type_names_raw.append(str(cell_type) if not pd.isna(cell_type) and str(cell_type) else str(getattr(row, "superclass")))
    type_names = sorted(set(type_names_raw))
    type_to_id = {name: i for i, name in enumerate(type_names)}
    node_type = np.asarray([type_to_id[name] for name in type_names_raw], dtype=np.int32)

    pair_keys = np.stack([node_type[edge_pre], node_type[edge_post]], axis=1)
    unique_pairs, edge_pair = np.unique(pair_keys, axis=0, return_inverse=True)
    edge_pair = edge_pair.astype(np.int32)

    output_indices = np.flatnonzero(kept_visual["superclass"].to_numpy() == "visual_projection").astype(np.int32)
    if len(output_indices) == 0:
        raise RuntimeError(f"No visual_projection neurons reached within max_hops={max_hops}; increase --hops")

    seed_rows: list[int] = []
    seed_cols: list[int] = []
    seed_vals: list[float] = []
    coords: list[tuple[int, int]] = []
    for y, x, old_nodes in column_rows:
        new_nodes = [int(old_to_new[n]) for n in old_nodes if old_to_new[n] >= 0]
        if not new_nodes:
            continue
        c = len(coords)
        coords.append((y, x))
        seed_rows.extend(new_nodes)
        seed_cols.extend([c] * len(new_nodes))
        seed_vals.extend([1.0] * len(new_nodes))

    np.savez_compressed(
        out_dir / "visual_connectome.npz",
        node_ids=node_ids,
        edge_pre=edge_pre,
        edge_post=edge_post,
        edge_count=edge_count,
        edge_weight=edge_weight.astype(np.float32),
        edge_sign=edge_sign,
        edge_pair=edge_pair,
        node_type=node_type,
        pair_types=unique_pairs.astype(np.int32),
        output_indices=output_indices,
        seed_rows=np.asarray(seed_rows, dtype=np.int32),
        seed_cols=np.asarray(seed_cols, dtype=np.int32),
        seed_vals=np.asarray(seed_vals, dtype=np.float32),
        column_coords=np.asarray(coords, dtype=np.float32),
    )
    (out_dir / "type_names.json").write_text(json.dumps(type_names, indent=2) + "\n")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset": "MaleCNS v1.0",
                "eye": eye,
                "max_hops": max_hops,
                "neurons": int(len(node_ids)),
                "edges": int(len(edge_pre)),
                "raw_synapses": int(edge_count.sum()),
                "cell_types": int(len(type_names)),
                "type_pairs": int(len(unique_pairs)),
                "optic_columns": int(len(coords)),
                "visual_projection_outputs": int(len(output_indices)),
                "fixed": ["neuron identities", "directed topology", "synapse counts", "presynaptic transmitter sign"],
                "trainable_later": ["tau per cell type", "bias per cell type", "gain per source/destination cell-type pair", "depth readout"],
                "sign_policy": NT_SIGN,
                "unknown_or_modulatory_sign": 0.0,
                "weight_normalization": "published synapse count divided by total retained incoming count of postsynaptic neuron",
                "retina_mapping": "RGB luminance sampled on published optic-column lattice and injected into real R1-R6 bodies; camera-to-fly photometry is an engineered assumption",
                "sources": SOURCES,
            },
            indent=2,
        )
        + "\n"
    )
    print((out_dir / "meta.json").read_text())
