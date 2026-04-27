"""
src/graph.py  —  Graph backbone for Graph-Augmented MADDPG.

Communication-Abstraction pattern (milestone report §3.1):

    raw obs [B, N, D]
        ↓  ProximityGraphBuilder
    adj     [B, N, N]        (binary, self-loops included)
        ↓  GCNLayer × L
    embedding [B, N, H]      (drop-in for raw obs in actor / critic)

Design choices
--------------
* Pure PyTorch (no PyG / DGL): portable to any SLURM node with torch only.
* Dense batched adjacency [B, N, N]: negligible cost for N ≤ ~10 agents.
* Symmetric-normalized GCN (Kipf & Welling, ICLR 2017) as the backbone.
  GAT can be slotted in later by swapping GCNLayer → GATLayer.
* Single shared GNNEncoder instance wraps as a TensorDictModule so it
  integrates cleanly with TorchRL's TensorDictSequential.
* Encoder can be prepended to both the actor and critic chains so that
  gradients flow through the encoder from both loss terms.

Observation layout assumed for simple_tag_v3
--------------------------------------------
  [self_vel(2), self_pos(2), landmark_rel_pos(...), other_agents_rel_pos(...), ...]
  Absolute 2-D position lives at indices 2:4 (POS_SLICE).

Typical usage
-------------
    from src.graph import make_gnn_encoder

    # Create a shared encoder
    enc = make_gnn_encoder(obs_dim=16, hidden_dim=128, group="adversary",
                           n_layers=2, radius=None, device=device)

    # Prepend to actor / critic for the collection policy and DDPGLoss
    actor  = TensorDictSequential(enc, base_actor)
    critic = TensorDictSequential(enc, base_critic)   # same nn.Module → shared weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

# Indices of the agent's own 2-D absolute position in the observation vector.
# simple_tag_v3 layout: [self_vel(2), self_pos(2), ...]
POS_SLICE = slice(2, 4)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class ProximityGraphBuilder(nn.Module):
    """Builds a batched binary adjacency matrix from agent positions.

    Nodes  = agents (rows / cols of the N×N matrix)
    Edges  = undirected pair (i,j) when ||pos_i − pos_j|| ≤ radius
    Self-loops are always added (required for the GCN normalisation).

    Args:
        radius: connectivity radius in world units.
                ``None`` → fully-connected graph (every pair gets an edge).
    """

    def __init__(self, radius: float | None = None):
        super().__init__()
        self.radius = radius

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: ``[B, N, obs_dim]`` — batched agent observations.
        Returns:
            adj: ``[B, N, N]`` — float binary adjacency (self-loops included).
        """
        pos = obs[..., POS_SLICE]                           # [B, N, 2]
        diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)        # [B, N, N, 2]
        dist = diff.norm(dim=-1)                            # [B, N, N]

        if self.radius is not None:
            adj = (dist <= self.radius).to(obs.dtype)
        else:
            adj = torch.ones_like(dist)

        # Self-loops: ensure every node aggregates its own features
        eye = torch.eye(obs.shape[-2], device=obs.device, dtype=obs.dtype)
        adj = (adj + eye.unsqueeze(0)).clamp(max=1.0)      # [B, N, N]
        return adj


# ---------------------------------------------------------------------------
# GCN message-passing
# ---------------------------------------------------------------------------

def _sym_normalize(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric degree normalisation: Â = D^{-½} A D^{-½}.

    Args:
        adj: ``[B, N, N]`` raw adjacency (self-loops already present).
    Returns:
        a_hat: ``[B, N, N]`` normalised adjacency.
    """
    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)    # [B, N, 1]
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt * adj * d_inv_sqrt.transpose(-1, -2)


class GCNLayer(nn.Module):
    """One GCN message-passing step (Kipf & Welling, ICLR 2017):

        H' = σ( Â · H · W + b )

    where  Â = D^{-½} A D^{-½}.

    All agents share the single weight matrix W  (permutation invariant).
    """

    def __init__(self, in_dim: int, out_dim: int,
                 activation: nn.Module | None = None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU() if activation is None else activation

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   ``[B, N, in_dim]``   node feature matrix.
            adj: ``[B, N, N]``        raw adjacency (self-loops included).
        Returns:
            h':  ``[B, N, out_dim]``  updated node features.
        """
        a_hat = _sym_normalize(adj)                         # [B, N, N]
        # aggregate neighbours, then project  (Â H W ≡ Â (H W) by linearity)
        return self.act(self.linear(torch.bmm(a_hat, h)))   # [B, N, out_dim]


# ---------------------------------------------------------------------------
# Multi-layer GNN encoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """Multi-layer GCN backbone for a group of agents.

    Processes all N agents' observations jointly:
    1. Builds a proximity graph on the fly from embedded positions.
    2. Runs L stacked GCN layers with shared weights.
    3. Returns per-agent relational embeddings that capture both local
       state and neighbourhood context.

    The output is a drop-in replacement for raw observations in the
    actor and critic networks.  Permutation invariance is guaranteed by
    the shared weight matrices inside each GCNLayer.

    Args:
        obs_dim:    dimensionality of the raw observation vector.
        hidden_dim: output embedding dimension (= actor / critic input dim).
        n_layers:   number of stacked GCN layers (default 2).
        radius:     proximity radius for edge construction.
                    ``None`` → fully-connected (recommended when N ≤ 5).
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
        """
        Args:
            obs: ``[B, N, obs_dim]``
        Returns:
            emb: ``[B, N, hidden_dim]``
        """
        adj = self.graph_builder(obs)   # [B, N, N]  — rebuilt every step
        h = obs
        for layer in self.layers:
            h = layer(h, adj)
        return h                        # [B, N, hidden_dim]


# ---------------------------------------------------------------------------
# TorchRL factory
# ---------------------------------------------------------------------------

def make_gnn_encoder(
    obs_dim: int,
    hidden_dim: int,
    group: str,
    n_layers: int = 2,
    radius: float | None = None,
    device: torch.device | None = None,
) -> TensorDictModule:
    """Wrap a :class:`GNNEncoder` as a TorchRL ``TensorDictModule``.

    The module reads the raw agent observations and writes per-agent
    relational embeddings:

        (group, "observation")  [B, N, obs_dim]   →   (group, "embedding")  [B, N, hidden_dim]

    Prepend one instance to the actor and **the same instance** to the
    critic so both networks share encoder weights and gradients from
    both loss terms propagate through the encoder:

    .. code-block:: python

        enc = make_gnn_encoder(obs_dim, hidden_dim, "adversary", device=device)
        actor  = TensorDictSequential(enc, base_actor)
        critic = TensorDictSequential(enc, base_critic)
        # enc is the same Python object → shared parameters

    Args:
        obs_dim:    raw observation dimension per agent.
        hidden_dim: output embedding dimension.
        group:      agent group name (e.g. ``"adversary"`` or ``"agent"``).
        n_layers:   number of GCN layers (default 2).
        radius:     proximity radius for edge construction (``None`` = fully connected).
        device:     torch device to place the encoder on.

    Returns:
        A ``TensorDictModule`` that in-place adds the embedding key to
        any ``TensorDict`` it receives.
    """
    encoder = GNNEncoder(obs_dim, hidden_dim, n_layers=n_layers, radius=radius)
    if device is not None:
        encoder = encoder.to(device)
    return TensorDictModule(
        module=encoder,
        in_keys=[(group, "observation")],
        out_keys=[(group, "embedding")],
    )
