from typing import Dict, Iterator
import argparse

from stable_baselines3 import PPO  # type: ignore

default_config = argparse.Namespace(
    bottleneck_temperature=1.5,
    bottleneck_hard=False,
    pre_arch=[0x20, 0x20],
    post_arch=[0x20],
    policy_activation="tanh",
    eval_freq=5_000,
    total_timesteps=100_000,
    eval_steps=1_000,
    device="cpu",
    alg=PPO,
    n_steps=0x400,  # Was 0x80
    batch_size=0x100,
    learning_rate=3e-3,
    init_model_path=None,
    goal_radius=1.0,
    world_radius=9.0,
    # TODO Rename this since it is confusing
    max_step_scale=3.0,
    entropy_coef=0.0,
    gamma=0.9,
    rs_multiplier=0.0,
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
            "rs_multiplier": x,
            **base,
        }


def learning_rate() -> Iterator[Dict]:
    n = 400
    hi = -1
    lo = -4
    for rs_multiplier in [1e-4, 1]:
        for i in range(n):
            x = 10 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "rs_multiplier": rs_multiplier,
                "learning_rate": x,
            }


def low_learning_rate() -> Iterator[Dict]:
    base = {
        "total_timesteps": 500_000,
    }
    n = 100
    hi = -3
    lo = -7
    # for rs_multiplier in [1e-4, 1]:
    for rs_multiplier in [1]:
        for i in range(n):
            x = 10 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "rs_multiplier": rs_multiplier,
                "learning_rate": x,
                **base,
            }


def bottleneck_size() -> Iterator[Dict]:
    n = 400
    hi = 3
    lo = 10
    for rs_multiplier in [1e-4, 1]:
        prev_x = None
        for i in range(n):
            x = int(2 ** (lo + (hi - lo) * i / (n - 1)))
            # Because of squashing with int, skip over duplicates
            if x == prev_x:
                continue
            prev_x = x
            yield {
                "pre_arch": [0x20, x],
                "rs_multiplier": rs_multiplier,
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
            "rs_multiplier": x,
            **base,
        }


def temperature() -> Iterator[Dict]:
    n = 400
    hi = 1
    lo = -1
    for rs_multiplier in [1e-4, 1]:
        for i in range(n):
            x = 2 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "rs_multiplier": rs_multiplier,
                "bottleneck_temperature": x,
            }
