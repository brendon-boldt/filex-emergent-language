import sys
from typing import Any, Tuple, List
import argparse
from argparse import Namespace
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

_cfg = argparse.Namespace(
    device="cuda",
    n_proc=4,
    alg=PPO,
    n_steps=0x80,
    batch_size=0x100,
    learning_rate=3e-4,  # default 3-e4
    single_step=False,
    env_lsize=2,
    action_scale=2 ** 0,
    fe_out_size=0x40,
    fe_out_ratio=4,
    policy_net_arch=[0x40] * 2,
    eval_episodes=1000,
    eval_freq=1000,
    total_timesteps=500_000,
    reward_threshold=0.95,
)


def make_env(env_constructor, rank, seed=0):
    def _init():
        env = env_constructor()
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def full_test(
    base_dir: Path, cfg: argparse.Namespace, idx: int
) -> Tuple[bool, float, int]:
    log_dir = base_dir / f"run-{idx}"
    writer = SummaryWriter(log_dir=log_dir)
    # loggable_hparams = {
    #     k: v if v in (int, float, bool) else str(v) for k, v in vars(cfg).items()
    # }
    # writer.add_hparams(loggable_hparams, {})
    with (log_dir / "config.txt").open("w") as fo:
        fo.write(str(cfg))

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
    env_lam = lambda: E.Scalable(**env_kwargs)
    env = SubprocVecEnv([make_env(env_lam, i) for i in range(cfg.n_proc)])
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
    stopper_callback = callbacks.StopTrainingOnRewardThreshold(
        reward_threshold=cfg.reward_threshold, verbose=0
    )
    logging_callback = LoggingCallback(
        eval_env=env_eval,
        n_eval_episodes=cfg.eval_episodes,
        eval_freq=cfg.eval_freq,
        callback_on_new_best=stopper_callback,
        writer=writer,
        verbose=0,
    )
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[logging_callback],
    )
    success = logging_callback.best_mean_reward > cfg.reward_threshold
    return success, logging_callback.best_mean_reward, model.num_timesteps


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("--num_trials", type=int, default=10)
    return parser.parse_args()


def run_trials(
    base_dir: Path, cfg: Namespace, name_props: List[str], num_trials: int
) -> None:
    name = "_".join(str(getattr(cfg, prop)) for prop in name_props)
    print(name)
    log_dir = base_dir / name
    if log_dir.exists():
        pass
    results = []
    for i in range(num_trials):
        results.append(full_test(log_dir, cfg, i))
        with (log_dir / "results.csv").open("a") as fo:
            fo.write(",".join(str(x) for x in results[-1]))
            fo.write("\n")


def main() -> None:
    args = get_args()

    exper_dir = Path("runs") / args.run_name
    if exper_dir.exists():
        raise ValueError()
    for action_lscale in (0, 1, 2, 3):
        for env_lsize in (2, 3, 4):
            if action_lscale >= env_lsize:
                continue
            cfg = Namespace(**vars(_cfg))
            cfg.action_scale = 2 ** action_lscale
            cfg.env_lsize = env_lsize
            run_trials(exper_dir, cfg, ["env_lsize", "action_scale"], args.num_trials)


if __name__ == "__main__":
    main()
