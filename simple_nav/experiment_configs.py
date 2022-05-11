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
import gym  # type: ignore

from . import env
from .util import log_range

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
    n_dims=None,
    n_opts=None,
)

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

def buffer_size_nd() -> Iterator[Dict]:
    base = {
        "environment": env.NoDynamics,
    }
    for x in log_range(2**0, 2**15, 100):
        # Avoid an issue with PPO
        if int(x) % default_config.batch_size == 1:
            x += 1
        yield {
            "n_steps": int(x),
            # Prevent name collisions due to int(x)
            "note": x,
            **base,
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


def buffer_size_test() -> Iterator[Dict]:
    for x in log_range(2**3, 2**15, 20):
        # Avoid an issue with PPO
        if int(x) % default_config.batch_size == 1:
            x += 1
        yield {
            "n_steps": int(x),
            # Prevent name collisions due to int(x)
            "note": x,
            "eval_freq": 100,
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
    for x in log_range(0.1, 10, 100):
        yield {
            "bottleneck_temperature": x,
        }


def rc_learning_rate() -> Iterator[Dict]:
    base = {
            "environment": env.Reconstruction,
            "n_dims": 1,
            "total_timesteps": 200_000,
            }
    for x in log_range(1e-6, 1e-1, 200):
        yield {
            "learning_rate": x,
            **base,
        }

def rc_time_steps() -> Iterator[Dict]:
    base = {
            "environment": env.Reconstruction,
            "n_dims": 1,
            # "total_timesteps": 200_000,
            }
    # for x in log_range(1e2, 1e7, 200):
    for x in log_range(1e2, 1e6, 100):
        ef = min(default_config.eval_freq, x // 5)
        yield {
            "total_timesteps": x,
            "eval_freq": ef,
            **base,
        }

def rc_buffer_size() -> Iterator[Dict]:
    base = {
            "environment": env.Reconstruction,
            "n_dims": 1,
            # "total_timesteps": 100_000 // 0x100,
            }
    # Keep the number of gradient steps constant
    timesteps_base = 100_000 // 0x100
    # for x in log_range(2**3, 2**15, 600):
    for x in log_range(2**3, 2**10, 200):
        # Avoid an issue with PPO
        if int(x) % default_config.batch_size == 1:
            x += 1
        timesteps = int(x) * timesteps_base
        eval_freq = min(default_config.eval_freq, timesteps // 5)
        yield {
            "n_steps": int(x),
            "total_timesteps": timesteps,
            "eval_freq":  eval_freq,
            # Prevent name collisions due to int(x)
            "note": x,
            **base,
        }

def rc_lexicon_size() -> Iterator[Dict]:
    base = {
            "environment": env.Reconstruction,
            "n_dims": 1,
            "total_timesteps": 200_000,
            }
    for x in log_range(2**8, 2**3, 400):
        yield {
            "pre_bottleneck_arch": [0x20, int(x)],
            "note": x,
            **base,
        }

def rc_temperature() -> Iterator[Dict]:
    base = {
            "environment": env.Reconstruction,
            "n_dims": 1,
            "total_timesteps": 200_000,
            }
    for x in log_range(0.1, 10, 100):
        yield {
            "bottleneck_temperature": x,
            **base,
        }

def scratch() -> Iterator[Dict]:
    base = {
            "environment": env.Signal,
            "n_dims": 5,
            "n_opts": 2,
            "eval_freq": 0x100,
            "total_timesteps": 200_000,
            }
    for x in range(20):
        yield {
            "note": x,
            **base,
        }

sg_base_cfg = {
            "environment": env.Signal,
            "n_dims": 5,
            "n_opts": 2,
            "total_timesteps": 200_000,
            }

def sg_learning_rate() -> Iterator[Dict]:
    base = sg_base_cfg
    for x in log_range(1e-6, 1e-1, 100):
        yield {
            "learning_rate": x,
            **base,
        }

def sg_temperature() -> Iterator[Dict]:
    base = sg_base_cfg
    for x in log_range(0.1, 10, 100):
        yield {
            **base,
            "bottleneck_temperature": x,
        }

def sg_time_steps() -> Iterator[Dict]:
    base = sg_base_cfg
    # for x in log_range(1e2, 1e7, 200):
    for x in log_range(1e2, 1e6, 100):
        ef = min(default_config.eval_freq, x // 5)
        yield {
            **base,
            "total_timesteps": x,
            "eval_freq": ef,
        }


def sg_buffer_size() -> Iterator[Dict]:
    base = sg_base_cfg
    timesteps_base = 100_000 // 0x100
    # for x in log_range(2**3, 2**15, 600):
    for x in log_range(2**3, 2**10, 100):
        # Avoid an issue with PPO
        if int(x) % default_config.batch_size == 1:
            x += 1
        timesteps = int(x) * timesteps_base
        eval_freq = min(default_config.eval_freq, timesteps // 5)
        yield {
            **base,
            "n_steps": int(x),
            "total_timesteps": timesteps,
            "eval_freq":  eval_freq,
            # Prevent name collisions due to int(x)
            "note": x,
        }

def sg_lexicon_size() -> Iterator[Dict]:
    base = sg_base_cfg
    for x in log_range(2**8, 2**3, 100):
        yield {
            **base,
            "pre_bottleneck_arch": [0x20, int(x)],
            "note": x,
        }
