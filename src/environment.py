import gymnasium as gym


def create_environment(name="FrozenLake-v1"):
    env = gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    is_slippery=True,
    success_rate=1.0/3.0,
    reward_schedule=(10, -1, 0)
)
    return env