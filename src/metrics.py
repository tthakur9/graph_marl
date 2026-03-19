"""
Implements proposal evaluation metrics computed from collector TensorDicts each iteration:
  - episode_return : mean total reward per episode per group
  - capture_rate : fraction of episodes in which at least one capture occurred
  - time_to_capture : mean step at which first capture occurred among captured episodes
  - collision_rate : mean predator-predator collisions per step
  - coverage_efficiency : mean pairwise distance between chasers (higher = better spread)

All functions accept the raw TensorDict `td` returned by the collector (shape [frames_per_batch]) 
and the group map from the environment.

Usage in train.py:
    from src.metrics import compute_metrics, format_metrics
    metrics = compute_metrics(td, base_env.group_map, n_agents, cfg)
    print(format_metrics(iteration, total_frames, metrics))
"""

from __future__ import annotations
from typing import Any
import torch
from tensordict import TensorDictBase


def _episode_mask(td: TensorDictBase) -> torch.Tensor:
    """
    Boolean mask [T] that is True on the last step of every episode.
    Uses the global `done` flag written by the collector.
    """
    done = td.get("next").get("done")
    return done.squeeze(-1).bool()


def episode_return(td: TensorDictBase, group: str) -> float | None:
    """
    Mean episode return for `group` over all completed episodes in this batch.
    RewardSum writes cumulative return under (group, "episode_reward") and
    resets it on the step AFTER done, so we read it on done steps.
    Returns None if no episode completed in this batch.
    """
    key = (group, "episode_reward")
    if key not in td.keys(include_nested=True):
        return None
    ep_reward = td[key]             # [T, n_agents, 1]
    mask = _episode_mask(td)        # [T]
    if not mask.any():
        return None
    # mean over agents then over episodes
    return ep_reward[mask].mean().item()


def capture_rate(td: TensorDictBase) -> float | None:
    """
    Fraction of completed episodes in which at least one capture occurred.
    A capture is detected when an adversary receives a positive reward spike
    (the +10 contact reward in simple_tag).  We flag any episode whose
    max per-step adversary reward exceeds a threshold as a capture episode.

    Returns None if no episode completed.
    """
    CAPTURE_THRESHOLD = 5.0 # simple_tag awards +10 on contact

    done = td.get("next").get("done").squeeze(-1).bool() # [T]
    if not done.any():
        return None

    # adversary reward: [T, n_adv, 1] → max over agents → [T]
    next_td = td.get("next")
    adv_rew = next_td.get(("adversary", "reward"))  # [T, n_adv, 1]
    max_rew = adv_rew.amax(dim=(-2, -1)) # [T]

    # walk episodes - a done at step t ends the episode [0..t]
    episode_captured = []
    ep_max = []
    for t in range(len(done)):
        ep_max.append(max_rew[t].item())
        if done[t]:
            episode_captured.append(max(ep_max) >= CAPTURE_THRESHOLD)
            ep_max = []

    if not episode_captured:
        return None
    return float(sum(episode_captured)) / len(episode_captured)


def time_to_capture(td: TensorDictBase, max_steps: int) -> float | None:
    CAPTURE_THRESHOLD = 5.0
    done    = td.get("next").get("done").squeeze(-1).bool()   # fix: read from next
    next_td = td.get("next")
    adv_rew = next_td.get(("adversary", "reward")).amax(dim=(-2, -1))

    first_capture_steps = []   # only first capture per episode
    ep_start = 0
    ep_captured = False
    for t in range(len(done)):
        if adv_rew[t].item() >= CAPTURE_THRESHOLD and not ep_captured:
            first_capture_steps.append((t - ep_start) / max(max_steps, 1))
            ep_captured = True
        if done[t]:
            ep_start   = t + 1
            ep_captured = False

    if not first_capture_steps:
        return None
    return float(sum(first_capture_steps)) / len(first_capture_steps)


def collision_rate(td: TensorDictBase) -> float | None:
    """
    Mean predator-predator collisions per step, approximated from observations.
    simple_tag_v3 includes relative positions of other agents in each agent's
    observation vector.  Two agents are considered colliding when their
    estimated relative distance < collision_radius.

    Observation layout for adversaries (dim 16):
      [0:2] self velocity
      [2:4] self position
      [4:6] landmark 0 relative pos
      [6:8] landmark 1 relative pos
      [8:10] other adversary 0 relative pos
      [10:12] other adversary 1 relative pos   (only when n_adv >= 3)
      [12:14] agent 0 relative pos
      [14:16] agent 0 velocity

    We read pairwise relative positions between chasers and count pairs
    whose L2 distance < COLLISION_RADIUS.

    Returns None if observation layout cannot be parsed (n_adv != 3).
    """
    COLLISION_RADIUS = 0.15 # tuned for simple_tag unit scale
    N_LANDMARKS = 2

    obs = td.get(("adversary", "observation"))  # [T, n_adv, obs_dim]
    T, n_adv, obs_dim = obs.shape
    if n_adv != 3:
        # layout varies with n_adv — skip rather than mis-parse
        return None

    # relative positions of other adversaries start after
    # self_vel(2) + self_pos(2) + landmark_rel_pos(2*N_LANDMARKS)
    offset = 2 + 2 + 2 * N_LANDMARKS  # = 8

    # agent i sees the other (n_adv - 1) adversaries starting at offset
    # For n_adv=3: agent 0 sees agents 1,2 at obs[8:10] and [10:12]
    # We reconstruct absolute positions from self_pos + relative_pos
    self_pos = obs[:, :, 2:4] # [T, n_adv, 2]

    collisions_per_step = []
    for t in range(T):
        count = 0
        for i in range(n_adv):
            for j_idx in range(n_adv - 1):
                rel_idx = offset + j_idx * 2
                rel_pos = obs[t, i, rel_idx: rel_idx + 2] # relative pos of neighbour
                dist    = rel_pos.norm().item()
                if dist < COLLISION_RADIUS:
                    count += 1
        # each collision counted twice (once per agent), divide by 2
        collisions_per_step.append(count / 2)

    return float(sum(collisions_per_step)) / max(T, 1)


def coverage_efficiency(td: TensorDictBase) -> float | None:
    """
    Mean pairwise L2 distance between chasers, averaged over all timesteps.
    Higher value = chasers are more spread out = better coverage of the arena.

    Reconstructed from self_pos in each adversary's observation.
    """
    obs = td.get(("adversary", "observation")) # [T, n_adv, obs_dim]
    T, n_adv, _ = obs.shape
    if n_adv < 2:
        return None

    self_pos = obs[:, :, 2:4] # [T, n_adv, 2]

    dists = []
    for i in range(n_adv):
        for j in range(i + 1, n_adv):
            d = (self_pos[:, i, :] - self_pos[:, j, :]).norm(dim=-1) # [T]
            dists.append(d)

    if not dists:
        return None
    return torch.stack(dists, dim=0).mean().item()


# Aggregator
def compute_metrics(
    td: TensorDictBase,
    group_map: dict[str, list[str]],
    n_agents: dict[str, int],
    max_steps: int,
) -> dict[str, Any]:
    """
    Compute all proposal metrics from one collector batch.
    Returns a flat dict of metric_name -> value (float or None).
    """
    groups = list(group_map.keys())
    metrics: dict[str, Any] = {}

    # per-group episode return
    for g in groups:
        metrics[f"ep_return_{g}"] = episode_return(td, g)

    # capture metrics (chaser perspective)
    metrics["capture_rate"]      = capture_rate(td)
    metrics["time_to_capture"]   = time_to_capture(td, max_steps)

    # predator spatial metrics
    metrics["collision_rate"]    = collision_rate(td)
    metrics["coverage_eff"]      = coverage_efficiency(td)

    return metrics


def format_metrics(iteration: int, total_frames: int, metrics: dict[str, Any]) -> str:
    """Format metrics dict into a single log line."""
    def _fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    parts = [f"iter={iteration:04d}", f"frames={total_frames:7d}"]
    for k, v in metrics.items():
        parts.append(f"{k}={_fmt(v)}")
    return "  ".join(parts)
