from pettingzoo.mpe import simple_tag_v3
from supersuit import pad_observations_v0, pad_action_space_v0

def make_env(seed=1, max_cycles=200, continuous_actions=True, pad=True):
    # Simple tag is predator-prey env commonly used in MADDPG
    env = simple_tag_v3.parallel_env(max_cycles=max_cycles, continuous_actions=continuous_actions)
    if pad:
        env = pad_observations_v0(env)
        env = pad_action_space_v0(env)
    env.reset(seed=seed)
    return env

if __name__ == "__main__":
    env = make_env()
    print("agents:", env.agents)
    a0 = env.agents[0]
    print("obs space:", env.observation_space(a0))
    print("act space:", env.action_space(a0))
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    print("step ok. rewards keys:", rewards.keys())


