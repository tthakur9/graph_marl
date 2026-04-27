from __future__ import annotations
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MultiAgentMLP, TanhDelta, ProbabilisticActor

# Centralized critic
def make_critic(
    obs_dim: int,
    act_dim: int,
    n_agents: int,
    hidden: int,
    device: torch.device,
    group: str,
    share_params: bool,
    feature_key: str = "observation",
) -> TensorDictModule:
    """Centralized critic - maps global obs+act -> group station action values"""
    in_dim = n_agents * (obs_dim + act_dim)
    mlp = MultiAgentMLP(
        n_agent_inputs=in_dim,
        n_agent_outputs=1,
        n_agents=n_agents,
        centralized=True,
        share_params=share_params,
        depth=2,
        num_cells=hidden,
        activation_class=nn.Tanh,
        device=device,
    )

    class CentralizedCriticNet(nn.Module):
        """Flattens global obs+act, tiles per agent, passes through MLP."""
        def __init__(self, n_agents, mlp):
            super().__init__()
            self.n_agents = n_agents
            self.mlp = mlp

        def forward(self, obs, act):
            # obs: [B, n, obs_dim], act: [B, n, act_dim]
            B = obs.shape[:-2]
            flat = torch.cat([obs, act], dim=-1).flatten(-2) # [B, n * (obs_dim + act_dim)]
            tiled = flat.unsqueeze(-2).expand(*B, self.n_agents, -1) # [B, n, (obs_dim + act_dim)]
            return self.mlp(tiled) # [B, n, 1]

    return TensorDictModule(
        module=CentralizedCriticNet(n_agents, mlp),
        in_keys=[(group, feature_key), (group, "action")],
        out_keys=["state_action_value"],
    )

# Decentralized actor
def make_actor(
    obs_dim: int,
    act_dim: int,
    n_agents: int,
    hidden: int,
    device: torch.device,
    group: str,
    share_params: bool,
    action_spec,
    feature_key: str = "observation",
) -> ProbabilisticActor:
    mlp = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=act_dim,
        n_agents=n_agents,
        centralized=False,
        share_params=share_params,
        depth=2,
        num_cells=hidden,
        activation_class=nn.Tanh,
        device=device,
    )
    actor_net = TensorDictModule(
        module=mlp,
        in_keys=[(group, feature_key)],
        out_keys=[(group, "param")],
    )
    return ProbabilisticActor(
        module=actor_net,
        spec=action_spec,
        in_keys=[(group, "param")],
        out_keys=[(group, "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "low":  action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=False,
    )
