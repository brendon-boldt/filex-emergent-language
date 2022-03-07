"""
Configurations for experiments

Configurations that are used in the paper are annotated with the comment PAPER

Each "configuration" is a function which generates an iterator of dictionaries.
Each dictionary should contain the hyperparameters that are different from the
default.  Note that if any yielded dictionary is exactly the same to another
one in the experiment, They will overwrite each other. This can be avoided by
adding a dummy parameter like `{"note": i}` for each iteration of the config
loop.

"""
from typing import Dict, Iterator
import argparse
import warnings

from stable_baselines3 import PPO  # type: ignore

from . import env

default_config = argparse.Namespace(
    environment=env.NavToCenter,
    bottleneck_temperature=1.5,
    bottleneck_hard=False,
    pre_bottleneck_arch=[2**5, 2**6],
    post_bottleneck_arch=[2**5],
    policy_activation="tanh",
    eval_freq=5_000,
    total_timesteps=50_000,
    eval_episodes_logging=200,
    eval_episodes=3_000,
    rl_algorithm=PPO,
    n_steps=0x100,
    batch_size=0x40,
    learning_rate=3e-3,
    init_model_path=None,
    goal_radius=1.0,
    world_radius=9.0,
    eval_world_radius=9.0,
    # Maximum step count as a multiplier of world_radius
    max_step_scale=3.0,
    gamma=0.9,
    sparsity=1,
    biased_reward_shaping=False,
)


def log_range(low: float, high: float, steps: int) -> Iterator[float]:
    for i in range(steps):
        yield low * (high / low) ** (i / (steps - 1))


def quick_test() -> Iterator[Dict]:
    base = {
        "total_timesteps": 40_000,
    }
    for x in log_range(1e-4, 0.1, 4):
        yield {
            "learning_rate": x,
            **base,
        }


def learning_rate() -> Iterator[Dict]:
    for x in log_range(1e-4, 0.1, 400):
        yield {
            "learning_rate": x,
        }


def buffer_size() -> Iterator[Dict]:
    for x in log_range(2**3, 2**15, 600):
        # Avoid an issue with PPO
        if int(x) % default_config.batch_size == 1:
            x += 1
        yield {
            "n_steps": int(x),
            # Prevent name collisions due to int(x)
            "note": x,
        }


def lexicon_size() -> Iterator[Dict]:
    for x in log_range(2**8, 2**3, 400):
        yield {
            "pre_bottleneck_arch": [0x20, int(x)],
            "note": x,
        }


def train_steps() -> Iterator[Dict]:
    for x in log_range(1_000_000, 100, 400):
        yield {
            "total_timesteps": int(x),
        }


def temperature() -> Iterator[Dict]:
    for x in log_range(0.5, 2, 400):
        yield {
            "bottleneck_temperature": x,
        }
