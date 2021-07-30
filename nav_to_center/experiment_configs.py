from typing import Dict, Iterator
import argparse

from stable_baselines3 import PPO  # type: ignore

default_config = argparse.Namespace(
    bottleneck_temperature=1.5,
    bottleneck_hard=False,
    pre_bottleneck_arch=[0x20, 0x20],
    post_bottleneck_arch=[0x20],
    policy_activation="tanh",
    eval_freq=5_000,
    total_timesteps=100_000,
    eval_episodes_logging=200,
    eval_episodes=3_000,
    rl_algorithm=PPO,
    n_steps=0x400,  # Was 0x80
    batch_size=0x100,
    learning_rate=3e-3,
    init_model_path=None,
    goal_radius=1.0,
    world_radius=9.0,
    # Maximum step count as a multiplier of world_radius
    max_step_scale=3.0,
    gamma=0.9,
    sparsity=float("inf"),
)


def quick_test() -> Iterator[Dict]:
    base = {
        "total_timesteps": 40_000,
    }
    n = 4
    hi = 0
    lo = -4
    for i in range(n):
        x = 10 ** (lo + (hi - lo) * i / (n - 1))
        yield {
            "sparsity": x,
            **base,
        }


def learning_rate() -> Iterator[Dict]:
    n = 400
    hi = -1
    lo = -4
    for sparsity in [1e-4, 1]:
        for i in range(n):
            x = 10 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "sparsity": sparsity,
                "learning_rate": x,
            }


def low_learning_rate() -> Iterator[Dict]:
    base = {
        "total_timesteps": 500_000,
    }
    n = 100
    hi = -3
    lo = -7
    # for sparsity in [1e-4, 1]:
    for sparsity in [1]:
        for i in range(n):
            x = 10 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "sparsity": sparsity,
                "learning_rate": x,
                **base,
            }


def bottleneck_size() -> Iterator[Dict]:
    n = 400
    hi = 3
    lo = 10
    for sparsity in [1e-4, 1]:
        prev_x = None
        for i in range(n):
            x = int(2 ** (lo + (hi - lo) * i / (n - 1)))
            # Because of squashing with int, skip over duplicates
            if x == prev_x:
                continue
            prev_x = x
            yield {
                "pre_bottleneck_arch": [0x20, x],
                "sparsity": sparsity,
            }


def sparsity() -> Iterator[Dict]:
    base = {
        "total_timesteps": 40_000,
    }
    n = 1000
    hi = 0
    lo = -4
    for i in range(n):
        x = 10 ** (lo + (hi - lo) * i / (n - 1))
        yield {
            "sparsity": x,
            **base,
        }


def temperature() -> Iterator[Dict]:
    n = 400
    hi = 1
    lo = -1
    for sparsity in [1e-4, 1]:
        for i in range(n):
            x = 2 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "sparsity": sparsity,
                "bottleneck_temperature": x,
            }
