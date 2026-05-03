"""
TorchRL MADDPG baseline on PettingZoo MPE Simple Tag.

  - MADDPG is implemented by running one DDPGLoss per agent group.
  - Each group gets its own actor, critic, DDPGLoss, SoftUpdate, and optimizers.
  - The centralized critic is built manually as a TensorDictModule that
    reads concatenated global observations + actions and writes a per-agent value estimate.
  - A single SyncDataCollector drives collection, where both actors are composed into
    one TensorDictSequential with AdditiveGaussianModule noise for exploration.
  - One shared ReplayBuffer stores full multi-agent TensorDicts.
  - Hyperparameters are loaded from conf/baseline.yaml via OmegaConf.
  - Proposal metrics (capture rate, time-to-capture, collision rate, coverage
    efficiency) are computed each iteration via src/metrics.py.
"""

from __future__ import annotations
import argparse
import csv
import datetime
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.envs import PettingZooEnv, RewardSum, TransformedEnv, check_env_specs
from torchrl.modules import AdditiveGaussianModule
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from src.models import make_actor, make_critic
from src.metrics import compute_metrics, format_metrics
from src.evaluate import run_eval
from src.graph import make_gnn_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    args = parser.parse_args()

    # Config
    cfg_path = args.config if args.config else Path(__file__).parent.parent / "conf" / "graph.yaml"
    cfg = OmegaConf.load(cfg_path)
    if args.seed is not None:
        cfg.seed = args.seed

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(cfg))

    total_frames_target = cfg.collection.frames_per_batch * cfg.collection.n_iters

    # Output directory for this run
    run_tag = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{cfg.seed}"
    run_group = cfg.get("run_group", cfg.graph.backbone)
    runs_dir = Path(__file__).parent.parent / "runs" / run_group / run_tag
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runs_dir / "metrics.csv"
    print(f"Run dir: {runs_dir}")

    # Environment
    base_env = PettingZooEnv(
        task=cfg.env.task,
        parallel=True,
        seed=cfg.seed,
        continuous_actions=True,
        num_good=cfg.env.n_evaders,
        num_adversaries=cfg.env.n_chasers,
        num_obstacles=cfg.env.n_obstacles,
        max_cycles=cfg.env.max_steps,
    )
    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"] * len(base_env.group_map),
        ),
    )
    check_env_specs(env)

    # Separate env for deterministic eval (avoids interfering with the collector)
    eval_base = PettingZooEnv(
        task=cfg.env.task,
        parallel=True,
        seed=cfg.seed + 1,
        continuous_actions=True,
        num_good=cfg.env.n_evaders,
        num_adversaries=cfg.env.n_chasers,
        num_obstacles=cfg.env.n_obstacles,
        max_cycles=cfg.env.max_steps,
    )
    eval_env = TransformedEnv(
        eval_base,
        RewardSum(
            in_keys=eval_base.reward_keys,
            reset_keys=["_reset"] * len(eval_base.group_map),
        ),
    )

    groups   = list(base_env.group_map.keys())
    n_agents = {g: len(base_env.group_map[g]) for g in groups}
    obs_dims = {g: env.observation_spec[g, "observation"].shape[-1] for g in groups}
    act_dims = {g: env.full_action_spec[g, "action"].shape[-1]      for g in groups}
    share_params = {g: True for g in groups}

    print("Groups:", groups)
    for g in groups:
        print(f"  {g}: n={n_agents[g]}, obs={obs_dims[g]}, act={act_dims[g]}")

    encoders = {
        g: make_gnn_encoder(obs_dims[g], cfg.architecture.hidden, g,
                            n_layers=cfg.graph.n_layers,
                            radius=cfg.graph.radius,
                            backbone=cfg.graph.backbone,
                            n_heads=cfg.graph.n_heads,
                            device=device)
        for g in groups
    }

    # Actors
    base_actors = {
        g: make_actor(cfg.architecture.hidden, act_dims[g], n_agents[g],
                    cfg.architecture.hidden, device, g, share_params[g],
                    env.full_action_spec_unbatched[g, "action"],
                    feature_key="embedding")
        for g in groups
    }
    actors = {g: TensorDictSequential(encoders[g], base_actors[g]) for g in groups}

    actors_explore = {
        g: TensorDictSequential(
            actors[g],
            AdditiveGaussianModule(
                spec=env.full_action_spec[g, "action"],
                annealing_num_steps=total_frames_target // 2,
                action_key=(g, "action"),
                device=device,
            ),
        )
        for g in groups
    }
    collector_policy = TensorDictSequential(*[actors_explore[g] for g in groups])

    # Critics
    base_critics = {
        g: make_critic(cfg.architecture.hidden, act_dims[g], n_agents[g],
                    cfg.architecture.hidden, device, g, share_params[g],
                    feature_key="embedding")
        for g in groups
    }
    critics = {g: TensorDictSequential(encoders[g], base_critics[g]) for g in groups}

    # Loss modules — one DDPGLoss per group
    loss_modules    = {}
    target_updaters = {}
    for g in groups:
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
            state_action_value="state_action_value",
        )
        loss.make_value_estimator(ValueEstimators.TD0, gamma=cfg.maddpg.gamma)
        loss.to(device)
        loss_modules[g]    = loss
        target_updaters[g] = SoftUpdate(loss, eps=1 - cfg.maddpg.polyak_tau)

    # Optimizers
    actor_optims  = {g: torch.optim.Adam(actors[g].parameters(),  lr=cfg.training.lr) for g in groups}
    critic_optims = {g: torch.optim.Adam(critics[g].parameters(), lr=cfg.training.lr) for g in groups}

    # Replay buffer
    rb = ReplayBuffer(
        storage=LazyMemmapStorage(cfg.replay_buffer.memory_size),
        sampler=RandomSampler(),
        batch_size=cfg.training.batch_size,
    )

    # Collector
    collector = SyncDataCollector(
        env,
        policy=collector_policy,
        frames_per_batch=cfg.collection.frames_per_batch,
        total_frames=total_frames_target,
        device="cpu",
        storing_device="cpu",
    )

    # CSV: pre-define all columns so eval columns are always present
    train_metric_keys = [f"return_{g}" for g in groups]
    eval_metric_keys = [f"eval_ep_return_{g}" for g in groups]
    csv_fieldnames = ["iteration", "total_frames"] + train_metric_keys + eval_metric_keys
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, extrasaction="ignore")
    csv_writer.writeheader()

    # Training loop
    total_frames = 0

    for iteration, td in enumerate(collector):
        td = td.to(device)
        rb.extend(td.reshape(-1).cpu())
        total_frames += cfg.collection.frames_per_batch

        # Which groups train this iteration
        training_groups = groups.copy()
        if iteration >= cfg.training.stop_evader_iter and "agent" in training_groups:
            training_groups.remove("agent")

        if len(rb) < cfg.training.batch_size:
            metrics = compute_metrics(td, base_env.group_map)
            print(format_metrics(iteration, total_frames, metrics))
            continue

        for _ in range(cfg.training.n_opt_steps):
            batch = rb.sample().to(device)

            # Broadcast global done/terminated to per-agent shape for each group
            for g in groups:
                n = n_agents[g]
                for split in [batch, batch.get("next")]:
                    split.set(
                        (g, "done"),
                        split.get("done").unsqueeze(-2).expand(*split.batch_size, n, 1),
                    )
                    split.set(
                        (g, "terminated"),
                        split.get("terminated").unsqueeze(-2).expand(*split.batch_size, n, 1),
                    )

            for g in training_groups:
                # Critic step
                critic_loss_td = loss_modules[g](batch)
                critic_loss    = critic_loss_td["loss_value"]
                critic_optims[g].zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critics[g].parameters(), cfg.training.max_grad_norm)
                critic_optims[g].step()

                # Actor step
                actor_loss_td = loss_modules[g](batch)
                actor_loss    = actor_loss_td["loss_actor"]
                actor_optims[g].zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actors[g].parameters(), cfg.training.max_grad_norm)
                actor_optims[g].step()

                target_updaters[g].step()

        # Metrics & logging
        metrics = compute_metrics(td, base_env.group_map)
        print(format_metrics(iteration, total_frames, metrics))

        # Intermittent deterministic evaluation
        eval_metrics = {}
        if (iteration + 1) % cfg.eval.interval == 0:
            eval_metrics = run_eval(
                actors, eval_env, groups,
                cfg.eval.n_episodes, device, cfg.env.max_steps,
            )
            print("  EVAL  " + "  ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items()))

        # CSV logging
        row = {"iteration": iteration, "total_frames": total_frames, **metrics, **eval_metrics}
        csv_writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    eval_env.close()

    # Save model weights
    weights_path = runs_dir / "weights.pt"
    torch.save(
        {g: {"actor": actors[g].state_dict(), "critic": critics[g].state_dict()} for g in groups},
        weights_path,
    )
    print(f"Weights saved to {weights_path}")

    env.close()
    collector.shutdown()
    print("Training complete.")
    

if __name__ == "__main__":
    main()
