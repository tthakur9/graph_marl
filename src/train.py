"""
TorchRL MADDPG baseline on PettingZoo MPE Simple Tag.

  - MADDPG is implemented by running one DDPGLoss per agent group.
  - Each group gets its own actor, critic, DDPGLoss, SoftUpdate, and optimizers.
  - The centralized critic is built manually as a TensorDictModule that
    reads concatenated global obsservations + actions and writes a per-agent value estimate.
  - A single SyncDataCollector drives collection, where both actors are composed into
    one TensorDictSequential with AdditiveGaussianModule noise for exploration.
  - One shared ReplayBuffer stores full multi-agent TensorDicts.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.envs import PettingZooEnv, RewardSum, TransformedEnv, check_env_specs
from torchrl.modules import AdditiveGaussianModule
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from src.models import make_actor, make_critic

# Hyperparams
SEED = 0
torch.manual_seed(SEED)

# Environment
MAX_STEPS = 100  # episode length
N_CHASERS = 3  # adversaries
N_EVADERS = 1  # agents
N_OBSTACLES = 2

# Collection
FRAMES_PER_BATCH = 1_000 # env frames per collector iteration
N_ITERS = 150 # total collector iterations
TOTAL_FRAMES = FRAMES_PER_BATCH * N_ITERS

# Replay buffer
MEMORY_SIZE = 1_000_000

# Training
N_OPT_STEPS = 100 # gradient steps per collection iteration
TRAIN_BATCH_SIZE = 128
LR = 3e-4
MAX_GRAD_NORM = 1.0

# MADDPG
GAMMA = 0.99
POLYAK_TAU = 0.005 # soft target-network update coefficient

# Architecture
HIDDEN = 256

# Freeze evader training after this many iterations to let chasers catch up first
STOP_EVADER_ITER = N_ITERS // 2


# Main
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Environment
    base_env = PettingZooEnv(
        task="simple_tag_v3",
        parallel=True,
        seed=SEED,
        continuous_actions=True,
        num_good=N_EVADERS,
        num_adversaries=N_CHASERS,
        num_obstacles=N_OBSTACLES,
        max_cycles=MAX_STEPS,
    )
    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"] * len(base_env.group_map),
        ),
    )
    check_env_specs(env)

    groups = list(base_env.group_map.keys()) # ["adversary", "agent"]
    n_agents = {g: len(base_env.group_map[g]) for g in groups}
    obs_dims = {g: env.observation_spec[g, "observation"].shape[-1] for g in groups}
    act_dims = {g: env.full_action_spec[g, "action"].shape[-1] for g in groups}
    share_params = {g: True for g in groups}
    print("Groups:", groups)
    for g in groups:
        print(f"{g}: n={n_agents[g]}, obs={obs_dims[g]}, act={act_dims[g]}")

    # Actors
    actors = {
        g: make_actor(
            obs_dims[g], act_dims[g], n_agents[g],
            HIDDEN, device, g, share_params[g],
            action_spec=env.full_action_spec_unbatched[g, "action"],
        )
        for g in groups
    }

    # Gaussian exploration noise for collection
    actors_explore = {
        g: TensorDictSequential(
            actors[g],
            AdditiveGaussianModule(
                spec=env.full_action_spec[g, "action"],
                annealing_num_steps=TOTAL_FRAMES // 2,
                action_key=(g, "action"),
                device=device,
            ),
        ) 
        for g in groups
    }
    collector_policy = TensorDictSequential(*[actors_explore[g] for g in groups])

    # Critics
    critics = {
        g: make_critic(obs_dims[g], act_dims[g], n_agents[g], HIDDEN, device, g, share_params[g])
        for g in groups
    }

    # Loss modules — one DDPGLoss per group
    loss_modules = {}
    target_updaters = {}
    for g in groups:
        # CORRECT
        loss = DDPGLoss(
            actor_network=actors[g],
            value_network=critics[g],
            loss_function="l2",
            delay_value=True,
            delay_actor=False,
        )
        loss.set_keys(
            reward=(g, "reward"),
            done=(g, "done"),
            terminated=(g, "terminated"),
            state_action_value="state_action_value"
        )
        loss.make_value_estimator(ValueEstimators.TD0, gamma=GAMMA)
        loss.to(device)
        loss_modules[g] = loss
        target_updaters[g] = SoftUpdate(loss, eps=1-POLYAK_TAU)

    # Optimizers
    actor_optims = {g: torch.optim.Adam(actors[g].parameters(), lr=LR) for g in groups}
    critic_optims = {g: torch.optim.Adam(critics[g].parameters(), lr=LR) for g in groups}

    # Replay buffer
    rb = ReplayBuffer(
        storage=LazyMemmapStorage(MEMORY_SIZE),
        sampler=RandomSampler(),
        batch_size=TRAIN_BATCH_SIZE,
    )

    # Collector
    collector = SyncDataCollector(
        env,
        policy=collector_policy,
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=TOTAL_FRAMES,
        device=device,
        storing_device="cpu",
    )

    # Training loop
    total_frames = 0
    for iteration, td in enumerate(collector):
        # Flatten the time dimension before storing bc rb expects [B] tensordicts
        rb.extend(td.reshape(-1).cpu())
        total_frames += FRAMES_PER_BATCH

        # Determine who and if groups can train
        training_groups = groups.copy()
        if iteration >= STOP_EVADER_ITER and "agent" in training_groups:
            training_groups.remove("agent")
        if len(rb) < TRAIN_BATCH_SIZE:
            continue

        for _ in range(N_OPT_STEPS):
            batch = rb.sample().to(device)
            for g in groups:
                n = n_agents[g]
                for split in [batch, batch.get("next")]:
                    split.set((g, "done"), split.get("done").unsqueeze(-2).expand(*split.batch_size, n, 1))
                    split.set((g, "terminated"), split.get("terminated").unsqueeze(-2).expand(*split.batch_size, n, 1))

            for g in training_groups:
                critic_loss_td = loss_modules[g](batch)
                critic_loss = critic_loss_td["loss_value"]
                critic_optims[g].zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critics[g].parameters(), MAX_GRAD_NORM)
                critic_optims[g].step()

                actor_loss_td = loss_modules[g](batch)
                actor_loss = actor_loss_td["loss_actor"]
                actor_optims[g].zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actors[g].parameters(), MAX_GRAD_NORM)
                actor_optims[g].step()

                target_updaters[g].step()

        # Logging
        reward_parts = []
        for g in groups:
            key = (g, "episode_reward")
            if key in td.keys(include_nested=True):
                reward_parts.append(f"{g}={td[key].mean().item():.2f}")
        print(f"iter={iteration:04d}  frames={total_frames:7d}  " + "  ".join(reward_parts))

    env.close()
    collector.shutdown()
    print("Training complete.")


if __name__ == "__main__":
    main()
