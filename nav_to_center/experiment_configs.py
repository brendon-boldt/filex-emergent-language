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

sparsities = [1, 10_000]


def log_range(low: float, high: float, steps: int) -> Iterator[float]:
    for i in range(steps):
        yield low * (high / low) ** ((i - 1) / steps)


def quick_test() -> Iterator[Dict]:
    base = {
        "total_timesteps": 40_000,
    }
    for x in log_range(1e-4, 0.1, 4):
        yield {
            "sparsity": x,
            **base,
        }


def learning_rate() -> Iterator[Dict]:
    for sparsity in sparsities:
        for x in log_range(1e-4, 0.1, 400):
            yield {
                "sparsity": sparsity,
                "learning_rate": x,
            }


def bottleneck_size() -> Iterator[Dict]:
    for sparsity in sparsities:
        prev_x = None
        for x in log_range(2 ** 3, 2 ** 8, 400):
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
        "total_timesteps": 100_000,
    }
    for bottleneck_size in [32, 256]:
        for x in log_range(1, 10_000, 400):
            yield {
                "pre_bottleneck_arch": [0x20, bottleneck_size],
                "sparsity": x,
                **base,
            }


def temperature() -> Iterator[Dict]:
    for sparsity in sparsities:
        for x in log_range(0.5, 2, 400):
            yield {
                "sparsity": sparsity,
                "bottleneck_temperature": x,
            }


def world_radius() -> Iterator[Dict]:
    for sparsity in sparsities:
        for x in log_range(2, 20, 400):
            yield {
                "world_radius": x,
                "sparsity": sparsity,
            }
