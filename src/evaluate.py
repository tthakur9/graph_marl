"""
Deterministic evaluation of trained MADDPG actors.

Runs n_episodes with no exploration noise and returns mean episode return
per agent group — the primary comparable benchmark from the MADDPG paper.
"""

from __future__ import annotations
import torch
from tensordict.nn import TensorDictSequential


def run_eval(
    actors: dict,
    eval_env,
    groups: list[str],
    n_episodes: int,
    device: torch.device,
    max_steps: int,
) -> dict[str, float]:
    """
    Run n_episodes deterministically (no exploration noise).
    Returns mean episode return per group as eval_ep_return_<group>.
    """
    base_policy = TensorDictSequential(*[actors[g] for g in groups])

    def policy(td):
        return base_policy(td.to(device)).cpu()

    returns: dict[str, list[float]] = {g: [] for g in groups}

    with torch.no_grad():
        for _ in range(n_episodes):
            td = eval_env.rollout(max_steps=max_steps, policy=policy)
            for g in groups:
                # episode_reward is written by RewardSum; at the final step it holds
                # the total cumulative return. Shape: [T, n_agents, 1]
                ep_ret = td[(g, "episode_reward")][-1].mean().item()
                returns[g].append(ep_ret)

    return {f"eval_ep_return_{g}": sum(v) / len(v) for g, v in returns.items()}