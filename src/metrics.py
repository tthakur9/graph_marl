"""
src/metrics.py

Standard MARL evaluation metric: episode return per agent group.
Computed during deterministic eval rollouts (no exploration noise).
Comparable to MADDPG paper (Lowe et al. 2017) and BenchMARL results.
"""

from __future__ import annotations
from typing import Any
import torch
from tensordict import TensorDictBase


def episode_return(td: TensorDictBase, group: str) -> float | None:
    """
    Mean episode return for a group over all completed episodes in the batch.
    Reads (group, "episode_reward") written by RewardSum on done steps.
    Returns None if no episode completed in this batch.
    """
    key = (group, "episode_reward")
    if key not in td.keys(include_nested=True):
        return None
    done = td.get("next").get("done").squeeze(-1).bool()
    if not done.any():
        return None
    ep_reward = td[key]          # [T, n_agents, 1]
    return ep_reward[done].mean().item()


def compute_metrics(
    td: TensorDictBase,
    group_map: dict[str, list[str]],
) -> dict[str, Any]:
    """Compute episode return for each group."""
    return {
        f"return_{g}": episode_return(td, g)
        for g in group_map.keys()
    }


def format_metrics(iteration: int, total_frames: int, metrics: dict[str, Any]) -> str:
    def _fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"
    parts = [f"iter={iteration:04d}", f"frames={total_frames:7d}"]
    parts += [f"{k}={_fmt(v)}" for k, v in metrics.items()]
    return "  ".join(parts)
