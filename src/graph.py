"""
Graph backbone for Graph-Augmented MADDPG

Communication-Abstraction pattern:
- raw obs [B, N, D] -> ProximityGraphBuilder -> adj [B, N, N] (binary, self-loops included) 
  -> GCNLayer x L -> embedding [B, N, H] (drop-in for raw obs in actor / critic)
- Dense batched adjacency [B, N, N].
- Symmetric-normalized GCN as the backbone.
- Single shared GNNEncoder instance wraps as a TensorDictModule so it
  integrates cleanly with TorchRL's TensorDictSequential.
- Encoder can be prepended to both the actor and critic chains so that
  gradients flow through the encoder from both loss terms.
- Observation layout assumed for simple_tag_v3:
  [self_vel(2), self_pos(2), landmark_rel_pos(...), other_agents_rel_pos(...), ...]
  Absolute 2-D position lives at indices 2:4 (POS_SLICE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

# Indices of the agent's own 2-D absolute position in the observation vector
POS_SLICE = slice(2, 4)

# Graph construction
class ProximityGraphBuilder(nn.Module):
    """
    Builds a batched binary adjacency matrix from agent positions.
    Nodes = agents (rows / cols of the NxN matrix)
    Edges = undirected pair (i,j) when ||pos_i - pos_j|| <= radius
    Self-loops are always added bc required for the GCN normalisation.
    If radius is None, then graph is FC graph
    """
    def __init__(self, radius: float | None = None):
        super().__init__()
        self.radius = radius

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pos = obs[..., POS_SLICE] # [B, N, 2]
        diff = pos.unsqueeze(-2) - pos.unsqueeze(-3) # [B, N, N, 2]
        dist = diff.norm(dim=-1) # [B, N, N]

        if self.radius is not None:
            adj = (dist <= self.radius).to(obs.dtype)
        else:
            adj = torch.ones_like(dist)

        # Self-loops ensure every node aggregates its own features
        eye = torch.eye(obs.shape[-2], device=obs.device, dtype=obs.dtype)
        adj = (adj + eye.unsqueeze(0)).clamp(max=1.0) # [B, N, N]
        return adj


# GCN message-passing
def _sym_normalize(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric degree normalization"""
    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)    # [B, N, 1]
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt * adj * d_inv_sqrt.transpose(-1, -2)


class GCNLayer(nn.Module):
    """One GCN message-passing step"""
    def __init__(self, in_dim: int, out_dim: int,
                 activation: nn.Module | None = None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU() if activation is None else activation

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a_hat = _sym_normalize(adj) # [B, N, N]
        # aggregate neighbours, then project
        return self.act(self.linear(torch.bmm(a_hat, h))) # [B, N, out_dim]


# Multi-layer GNN encoder
class GNNEncoder(nn.Module):
    """
    Multi-layer GCN backbone for a group of agents.
    Processes all N agents obs jointly:
    1. Builds a proximity graph on the fly from embedded positions.
    2. Runs L stacked GCN layers with shared weights.
    3. Returns per-agent relational embeddings that capture both local
       state and neighbourhood context.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        radius: float | None = None,
    ):
        super().__init__()
        self.graph_builder = ProximityGraphBuilder(radius=radius)

        dims = [obs_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # SyncDataCollector calls the policy one step at a time, so obs
        # arrives as [N, obs_dim] (no batch dim). bmm requires 3D, so
        # we add/remove the batch dim around the GCN computation.
        unbatched = obs.dim() == 2
        if unbatched:
            obs = obs.unsqueeze(0)      # [1, N, obs_dim]

        adj = self.graph_builder(obs)   # [B, N, N] — rebuilt every step
        h = obs
        for layer in self.layers:
            h = layer(h, adj)

        return h.squeeze(0) if unbatched else h


# TorchRL factory
def make_gnn_encoder(
    obs_dim: int,
    hidden_dim: int,
    group: str,
    n_layers: int = 2,
    radius: float | None = None,
    device: torch.device | None = None,
) -> TensorDictModule:
    """Wrap a GNNEncoder as a TorchRL TensorDictModule"""
    encoder = GNNEncoder(obs_dim, hidden_dim, n_layers=n_layers, radius=radius)
    if device is not None:
        encoder = encoder.to(device)
    return TensorDictModule(
        module=encoder,
        in_keys=[(group, "observation")],
        out_keys=[(group, "embedding")],
    )
