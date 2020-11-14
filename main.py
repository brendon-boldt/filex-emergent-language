import sys
from typing import Any, Tuple, List
import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3  # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore
from stable_baselines3.dqn import CnnPolicy  # type: ignore
from stable_baselines3.common import callbacks  # type: ignore
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecTransposeImage,
    DummyVecEnv,
)  # type: ignore
from stable_baselines3.common.cmd_util import make_vec_env  # type: ignore
from stable_baselines3.common.utils import set_random_seed  # type: ignore
import torch
import numpy as np  # type: ignore
from torch.utils.tensorboard import SummaryWriter

import env as E
import nn
from callback import LoggingCallback

cfg = argparse.Namespace(
    run_name=sys.argv[1],
    device="cuda",
    n_proc=4,
    alg=PPO,
    n_steps=0x80,
    batch_size=0x100,
    # learning_rate=1e-4, # default 3-e4
    learning_rate=3e-4,  # default 3-e4
    single_step=False,
    env_lsize=6,
    action_scale=2 ** 3,
    fe_out_size=0x40,
    fe_out_ratio=4,
    policy_net_arch=[0x40] * 2,
    eval_episodes=1000,
    eval_freq=500,
    total_timesteps=10_000_000,
)


def make_env(env_constructor, rank, seed=0):
    def _init():
        env = env_constructor()
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def full_test() -> None:
    print(cfg)
    log_dir = Path(f"runs/{cfg.run_name}")
    if log_dir.exists():
        raise ValueError()
    writer = SummaryWriter(log_dir=log_dir)
    # loggable_hparams = {
    #     k: v if v in (int, float, bool) else str(v) for k, v in vars(cfg).items()
    # }
    # writer.add_hparams(loggable_hparams, {})

    env_kwargs = {
        "lsize": cfg.env_lsize,
        "single_step": cfg.single_step,
        "action_scale": cfg.action_scale,
    }
    env_lam = lambda: VecTransposeImage(DummyVecEnv([lambda: E.Scalable(**env_kwargs)]))
    policy_kwargs = {
        "features_extractor_class": nn.ScalableCnn,
        "features_extractor_kwargs": {
            "out_size": cfg.fe_out_size,
            "ratio": cfg.fe_out_ratio,
        },
        "net_arch": cfg.policy_net_arch,
    }
    # if len(sys.argv) >= 2 and sys.argv[1] == "train":
    if True:
        env_lam = lambda: E.Scalable(**env_kwargs)
        env = SubprocVecEnv([make_env(env_lam, i) for i in range(cfg.n_proc)])
        # env = env_lam()
        env_eval = VecTransposeImage(
            DummyVecEnv([lambda: E.Scalable(eval_reward=True, **env_kwargs)])
        )

        model = cfg.alg(
            "CnnPolicy",
            env,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            policy_kwargs=policy_kwargs,
            verbose=0,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
        )
        model.learn(
            # total_timesteps=int(policy_steps + learning_starts),
            total_timesteps=cfg.total_timesteps,
            # log_interval=5_000,
            callback=[
                LoggingCallback(
                    eval_env=env_eval,
                    n_eval_episodes=cfg.eval_episodes,
                    eval_freq=cfg.eval_freq,
                    writer=writer,
                    verbose=0,
                )
            ],
        )
        model.save("model-save")
        # We can't use a vectorized env for eval
        mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=1000)
        print(mean_reward, std_reward)


if __name__ == "__main__":
    # test_2d()
    full_test()
