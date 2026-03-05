"""
TorchRL baseline run (MADDPG-style) on PettingZoo MPE Simple Tag.
"""

from __future__ import annotations

import numpy as np
import torch

from envs.simple_tag import make_env
from supersuit import pad_observations_v0, pad_action_space_v0

# TorchRL imports
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs import ParallelEnv, PettingZooEnv, TransformedEnv, Compose
from torchrl.envs.transforms import DoubleToFloat, RewardSum, StepCounter
from torchrl.modules import MultiAgentMLP
from torchrl.objectives.multiagent import MADDPGLoss
from torch.optim import Adam


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Wrap PettingZoo env in TorchRL env
    base_env = PettingZooEnv(
        make_env(
            seed=seed,
            max_cycles=200,
            continuous_actions=True,
            pad=True
        ),
        parallel=True
    )
    env = TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
            StepCounter(max_steps=200),
            RewardSum()
        )
    ).to(device)

    # Infer shapes
    obs_spec = env.observation_spec
    act_spec = env.action_spec
    print("observation_spec:", obs_spec)
    print("action_spec:", act_spec)

    # Policy networks (out-of-box multi-agent MLP)
    # MultiAgentMLP builds per-agent modules with shared or unshared params.
    # For baseline, use shared params for actors, critics in MADDPG loss are centralized.
    actor = MultiAgentMLP(
        in_features=obs_spec.shape[-1],
        out_features=act_spec.shape[-1],
        n_agents=env.n_agents,
        centralized=False,
        share_params=True,
        device=device
    )

    # TorchRL MADDPG loss module will build critics if provided
    loss_module = MADDPGLoss(
        actor_network=actor,
        action_spec=act_spec,
        delay_value=True
    ).to(device)

    # Replay & Collector
    rb = ReplayBuffer(
        storage=LazyMemmapStorage(200_000),
        sampler=RandomSampler(),
        batch_size=1024,
    )

    collector = SyncDataCollector(
        env,
        policy=actor, # policy used for collection
        frames_per_batch=1024,
        total_frames=50_000, # keep small for first run
        device=device,
        storing_device="cpu",
    )

    optim = Adam(loss_module.parameters(), lr=1e-3)

    # Train loop (minimal)
    for i, td in enumerate(collector):
        # td is a TensorDict batch of transitions
        rb.extend(td.cpu())

        if len(rb) < rb.batch_size:
            continue

        batch = rb.sample().to(device)
        loss_td = loss_module(batch)

        loss = loss_td["loss_actor"] + loss_td["loss_value"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        loss_module.update_target_networks()

        if i % 10 == 0:
            # RewardSum transform puts episodic return under something like ("episode_reward",)
            print(f"iter={i:04d} loss={loss.item():.4f}")

    env.close()


if __name__ == "__main__":
    main()
