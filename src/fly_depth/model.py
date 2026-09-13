from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class MaleCNSVisualModel(nn.Module):
    """Differentiable rate dynamics constrained by the real MaleCNS visual graph."""

    def __init__(
        self,
        processed_dir: Path,
        steps: int = 8,
        dt: float = 0.02,
        tau_init: float = 0.05,
        min_tau: float = 0.005,
        rate_max: float = 5.0,
        input_gain: float = 1.0,
    ) -> None:
        super().__init__()
        data = np.load(Path(processed_dir) / "visual_connectome.npz")
        self.n_nodes = int(len(data["node_ids"]))
        self.feature_dim = int(len(data["output_indices"]))
        self.n_types = int(data["node_type"].max()) + 1
        self.n_pairs = int(data["edge_pair"].max()) + 1
        self.steps = int(steps)
        self.dt = float(dt)
        self.min_tau = float(min_tau)
        self.rate_max = float(rate_max)
        self.input_gain = float(input_gain)

        self.register_buffer("edge_pre", torch.from_numpy(data["edge_pre"].astype(np.int64)))
        self.register_buffer("edge_post", torch.from_numpy(data["edge_post"].astype(np.int64)))
        self.register_buffer("edge_weight", torch.from_numpy(data["edge_weight"].astype(np.float32)))
        self.register_buffer("edge_sign", torch.from_numpy(data["edge_sign"].astype(np.float32)))
        self.register_buffer("edge_pair", torch.from_numpy(data["edge_pair"].astype(np.int64)))
        self.register_buffer("node_type", torch.from_numpy(data["node_type"].astype(np.int64)))
        self.register_buffer("output_indices", torch.from_numpy(data["output_indices"].astype(np.int64)))
        self.register_buffer("seed_rows", torch.from_numpy(data["seed_rows"].astype(np.int64)))
        self.register_buffer("seed_cols", torch.from_numpy(data["seed_cols"].astype(np.int64)))
        self.register_buffer("seed_vals", torch.from_numpy(data["seed_vals"].astype(np.float32)))

        coords = data["column_coords"].astype(np.float32)
        y = 2.0 * (coords[:, 0] - coords[:, 0].min()) / max(float(np.ptp(coords[:, 0])), 1.0) - 1.0
        x = 2.0 * (coords[:, 1] - coords[:, 1].min()) / max(float(np.ptp(coords[:, 1])), 1.0) - 1.0
        self.register_buffer("retina_grid", torch.from_numpy(np.stack([x, y], axis=1)).float())

        tau_target = max(tau_init - min_tau, 1e-4)
        self.raw_tau = nn.Parameter(torch.full((self.n_types,), _inverse_softplus(tau_target)))
        self.type_bias = nn.Parameter(torch.zeros(self.n_types))
        self.log_pair_gain = nn.Parameter(torch.zeros(self.n_pairs))

    def biological_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "tau": self.min_tau + F.softplus(self.raw_tau),
            "bias": self.type_bias,
            "pair_gain": torch.exp(self.log_pair_gain),
        }

    def sample_retina(self, images: torch.Tensor) -> torch.Tensor:
        # images: Bx3xHxW. This is a camera->compound-eye adapter, not learned biology.
        b = images.shape[0]
        grid = self.retina_grid.view(1, 1, -1, 2).expand(b, -1, -1, -1)
        sampled = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.mean(dim=1).squeeze(1)  # B x optic_columns

    def _seed_current(self, columns: torch.Tensor) -> torch.Tensor:
        indices = torch.stack([self.seed_rows, self.seed_cols])
        seed = torch.sparse_coo_tensor(
            indices,
            self.seed_vals,
            size=(self.n_nodes, columns.shape[1]),
            device=columns.device,
        ).coalesce()
        return torch.sparse.mm(seed, columns.T).T * self.input_gain

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        columns = self.sample_retina(images)
        external = self._seed_current(columns)
        state = torch.zeros(images.shape[0], self.n_nodes, device=images.device, dtype=images.dtype)

        tau = self.min_tau + F.softplus(self.raw_tau)
        tau_nodes = tau[self.node_type]
        bias_nodes = self.type_bias[self.node_type]
        gain = torch.exp(self.log_pair_gain)
        values = self.edge_weight * self.edge_sign * gain[self.edge_pair]
        matrix = torch.sparse_coo_tensor(
            torch.stack([self.edge_post, self.edge_pre]),
            values,
            size=(self.n_nodes, self.n_nodes),
            device=images.device,
        ).coalesce()

        for _ in range(self.steps):
            rates = self.rate_max * torch.tanh(F.relu(state) / self.rate_max)
            recurrent = torch.sparse.mm(matrix, rates.T).T
            state = state + (self.dt / tau_nodes) * (-state + recurrent + external + bias_nodes)

        rates = self.rate_max * torch.tanh(F.relu(state) / self.rate_max)
        return rates[:, self.output_indices]

    def regularization(self, tau_init: float) -> torch.Tensor:
        params = self.biological_parameters()
        tau_reg = torch.log(params["tau"] / tau_init).square().mean()
        gain_reg = self.log_pair_gain.square().mean()
        bias_reg = self.type_bias.square().mean()
        return tau_reg + gain_reg + bias_reg


class DepthHead(nn.Module):
    """Small trainable probe from late visual activity to metric depth."""

    def __init__(self, feature_dim: int, hidden: int, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = tuple(output_size)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.output_size[0] * self.output_size[1]),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(-1, 1, *self.output_size)


class FlyDepthModel(nn.Module):
    def __init__(self, brain: MaleCNSVisualModel, head: DepthHead) -> None:
        super().__init__()
        self.brain = brain
        self.head = head

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.brain(image))


def depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.clamp(0.1, 10.0)
    target = target.clamp(0.1, 10.0)
    valid = torch.isfinite(target) & (target > 0.1) & (target < 10.0)
    pred, target = pred[valid], target[valid]
    abs_rel = ((pred - target).abs() / target).mean()
    rmse = torch.sqrt(((pred - target) ** 2).mean())
    ratio = torch.maximum(pred / target, target / pred)
    delta1 = (ratio < 1.25).float().mean()
    return {"abs_rel": float(abs_rel), "rmse": float(rmse), "delta1": float(delta1)}
