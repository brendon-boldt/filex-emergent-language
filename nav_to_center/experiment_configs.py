from typing import Dict, Iterator
import argparse
from math import pi

from stable_baselines3 import PPO  # type: ignore

from . import env

default_config = argparse.Namespace(
    environment=env.NavToCenter,
    bottleneck_temperature=1.5,
    bottleneck_hard=False,
    pre_bottleneck_arch=[0x20, 0x40],
    post_bottleneck_arch=[0x20],
    policy_activation="tanh",
    eval_freq=5_000,
    total_timesteps=100_000,
    eval_episodes_logging=200,
    eval_episodes=3_000,
    rl_algorithm=PPO,
    n_steps=0x400,
    batch_size=0x100,
    learning_rate=3e-3,
    init_model_path=None,
    goal_radius=1.0,
    world_radius=9.0,
    eval_world_radius=9.0,
    # Maximum step count as a multiplier of world_radius
    max_step_scale=3.0,
    gamma=0.9,
    sparsity=float("inf"),
    biased_reward_shaping=0.0,
)

sparsities = [1, float("inf")]


def log_range(low: float, high: float, steps: int) -> Iterator[float]:
    for i in range(steps):
        yield low * (high / low) ** (i / (steps - 1))


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
        for x in log_range(2, 40, 100):
            yield {
                "world_radius": x,
                "sparsity": sparsity,
            }


def goal_radius() -> Iterator[Dict]:
    for sparsity in sparsities:
        for x in log_range(0.3, 8, 60):
            yield {
                "goal_radius": x,
                "sparsity": sparsity,
                "total_timesteps": 60_000,
                "pre_bottleneck_arch": [0x20, 0x80],
            }


def nav_to_edges() -> Iterator[Dict]:
    base = {
        "environment": env.NavToEdges,
        "world_radius": 8.0,
        "goal_radius": 8.0,
        "n_steps": 0x100,
        "batch_size": 0x100,
        "total_timesteps": 40_000,
    }
    for x in sparsities:
        yield {
            "sparsity": x,
            **base,
        }
    # yield {
    #     "biased_reward_shaping": True,
    #     "sparsity": 1,
    #     **base,
    # }


def buffer_size() -> Iterator[Dict]:
    for sparsity in sparsities:
        for x in log_range(2 ** 5, 2 ** 12, 200):
            yield {
                "n_steps": int(x),
                "batch_size": int(x),
                "sparsity": sparsity,
                "note": x,
                "total_timesteps": 200_000,
                "pre_bottleneck_arch": [0x20, 0x40],
            }


def train_steps() -> Iterator[Dict]:
    for sparsity in sparsities:
        for x in log_range(100_000, 1_000_000, 60):
            yield {
                "sparsity": sparsity,
                "total_timesteps": int(x),
                "pre_bottleneck_arch": [0x20, 0x40],
            }
