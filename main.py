import sys
from typing import Any, Tuple, List, Callable
import argparse
from argparse import Namespace
from pathlib import Path
import pickle as pkl

import gym  # type: ignore
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
import util

_cfg = argparse.Namespace(
    device="cpu",
    n_proc=4,  # default: 4
    alg=PPO,
    n_steps=0x400,  # Was 0x80
    batch_size=0x40,  # Was 0x100
    learning_rate=3e-4,  # default: 3-e4
    single_step=False,
    env_lsize=6,
    action_scale=2 ** 2,
    fe_out_size=0x10,
    fe_out_ratio=4,
    bottleneck="gsm",
    policy_net_arch=[0x40] * 0,  # default: [0x40] * 2,
    eval_episodes=500,
    entropy_samples=100,
    eval_freq=5000,
    total_timesteps=5_000_000,
    reward_threshold=0.95,
    max_step_scale=4.5,  # default: 2.5
    pixel_space=False,
    pre_arch=[0x10, 0x10],
    post_arch=[0x10],
)

cfg_test = Namespace(
    n_test_episodes=1000,
)


def make_env(env_constructor, rank, seed=0):
    def _init():
        env = env_constructor()
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def make_env_kwargs(cfg: Namespace) -> gym.Env:
    return {
        "pixel_space": cfg.pixel_space,
        "lsize": cfg.env_lsize,
        "single_step": cfg.single_step,
        "action_scale": cfg.action_scale,
        "max_step_scale": cfg.max_step_scale,
    }


def make_policy_kwargs(cfg: Namespace) -> gym.Env:
    return {
        "features_extractor_class": nn.ScalableCnn
        if cfg.pixel_space
        else nn.BottleneckPolicy,
        "features_extractor_kwargs": {
            "out_size": cfg.fe_out_size,
            "ratio": cfg.fe_out_ratio,
            "bottleneck": cfg.bottleneck,
            "pre_arch": cfg.pre_arch,
            "post_arch": cfg.post_arch,
        },
        "net_arch": cfg.policy_net_arch,
    }


def make_model(cfg: Namespace) -> Any:
    env_kwargs = make_env_kwargs(cfg)
    if cfg.pixel_space:
        env_lam: Callable = lambda: VecTransposeImage(
            DummyVecEnv([lambda: E.Scalable(**env_kwargs)])
        )
    else:
        # env_lam = lambda: DummyVecEnv([lambda: E.Scalable(**env_kwargs)])
        env_lam = lambda: E.Scalable(**env_kwargs)

    if cfg.n_proc > 1:
        env = SubprocVecEnv([make_env(env_lam, i) for i in range(cfg.n_proc)])
    else:
        env = env_lam()
    policy_kwargs = make_policy_kwargs(cfg)
    model = cfg.alg(
        "CnnPolicy" if cfg.pixel_space else "MlpPolicy",
        env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        policy_kwargs=policy_kwargs,
        verbose=0,
        learning_rate=cfg.learning_rate,
        device=cfg.device,
    )
    return model


def full_test(base_dir: Path, cfg: argparse.Namespace, idx: int) -> None:
    log_dir = base_dir / f"run-{idx}"
    writer = SummaryWriter(log_dir=log_dir)
    # loggable_hparams = {
    #     k: v if v in (int, float, bool) else str(v) for k, v in vars(cfg).items()
    # }
    # writer.add_hparams(loggable_hparams, {})
    with (log_dir / "config.txt").open("w") as text_fo:
        text_fo.write(str(cfg))
    with (log_dir / "config.pkl").open("wb") as binary_fo:
        pkl.dump(cfg, binary_fo)

    env_kwargs = make_env_kwargs(cfg)
    if cfg.pixel_space:
        env_eval: Any = VecTransposeImage(
            DummyVecEnv([lambda: E.Scalable(is_eval=True, **env_kwargs)])
        )
    else:
        env_eval = DummyVecEnv([lambda: E.Scalable(is_eval=True, **env_kwargs)])
        # env_eval = lambda: E.Scalable(is_eval=True, **env_kwargs)

    stopper_callback = callbacks.StopTrainingOnRewardThreshold(
        reward_threshold=cfg.reward_threshold, verbose=0
    )
    logging_callback = LoggingCallback(
        eval_env=env_eval,
        n_eval_episodes=cfg.eval_episodes,
        eval_freq=cfg.eval_freq,
        # callback_on_new_best=stopper_callback,
        writer=writer,
        verbose=0,
        entropy_samples=cfg.entropy_samples,
    )
    model = make_model(cfg)
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[logging_callback],
    )
    # success = logging_callback.best_mean_reward > cfg.reward_threshold
    # return success, logging_callback.best_mean_reward, model.num_timesteps


def collect_metrics(path: Path) -> Any:
    with (path / "config.pkl").open("rb") as fo:
        cfg = pkl.load(fo)
    env = E.Scalable(is_eval=True, **make_env_kwargs(cfg))
    model = make_model(cfg)
    model.load(path / "best.zip")
    policy = model.policy.cpu()
    features_extractor = model.policy.features_extractor.cpu()
    bottleneck_values = []
    steps_values = []
    for ep in range(cfg_test.n_test_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            obs_tensor = torch.Tensor(obs)
            with torch.no_grad():
                act = policy(obs_tensor)[0].numpy()
                bn = features_extractor.forward_bottleneck(obs_tensor).numpy()
            bottleneck_values.append(bn)
            obs, _, done, _ = env.step(act)
            steps += 1
        steps_values.append(steps)
    entropies = util.calc_entropies(np.stack(bottleneck_values))
    print(entropies)
    print(np.mean(steps_values))


def run_trials(
    base_dir: Path, cfg: Namespace, name_props: List[str], num_trials: int
) -> None:
    name = "_".join(str(getattr(cfg, prop)) for prop in name_props)
    print(name)
    log_dir = base_dir / name
    if log_dir.exists():
        pass
    # results = []
    for i in range(num_trials):
        full_test(log_dir, cfg, i)
        # results.append(full_test(log_dir, cfg, i))
        # with (log_dir / "results.csv").open("a") as fo:
        #     fo.write(",".join(str(x) for x in results[-1]))
        #     fo.write("\n")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("--num_trials", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.command == "test":
        collect_metrics(Path(args.target))
    elif args.command == "run":
        exper_dir = Path("runs") / args.target
        print(args.target)
        if exper_dir.exists():
            raise ValueError()

        for env_lsize in range(4, 8):
            for bn in "sm", "gsm":
                cfg = Namespace(**vars(_cfg))
                cfg.action_scale = 2 ** (env_lsize - 4)
                cfg.env_lsize = env_lsize
                cfg.bottleneck = bn
                run_trials(exper_dir, cfg, ["bottleneck", "env_lsize"], args.num_trials)


if __name__ == "__main__":
    main()
