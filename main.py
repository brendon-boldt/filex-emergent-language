import sys
from typing import Any, Tuple, List, Callable, Iterator, Union
import argparse
from argparse import Namespace
from pathlib import Path
import pickle as pkl
from functools import partial
from itertools import chain

import gym  # type: ignore
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3  # type: ignore
from stable_baselines3.dqn import CnnPolicy  # type: ignore
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecTransposeImage,
    DummyVecEnv,
)  # type: ignore
from stable_baselines3.common.utils import set_random_seed  # type: ignore
import torch
import numpy as np  # type: ignore
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed  # type: ignore
from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore

import env as E
import nn
from callback import LoggingCallback
import util

_cfg = argparse.Namespace(
    device="cpu",
    n_proc_alg=1,  # default: 4
    alg=PPO,
    n_steps=0x400,  # Was 0x80
    batch_size=0x100,  # Was 0x100
    learning_rate=3e-4,  # default: 3-e4
    single_step=False,
    env_lsize=6,
    action_scale=2 ** 2,
    fe_out_size=0x10,
    fe_out_ratio=4,
    bottleneck="gsm",
    policy_net_arch=[0x40] * 0,  # default: [0x40] * 2,
    eval_episodes=500,
    entropy_samples=400,
    eval_freq=20000,
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
        env_lam = lambda: E.Scalable(**env_kwargs)
    if cfg.n_proc_alg > 1:
        env = SubprocVecEnv([make_env(env_lam, i) for i in range(cfg.n_proc_alg)])
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


def do_run(base_dir: Path, cfg: argparse.Namespace, idx: int) -> None:
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
    logging_callback = LoggingCallback(
        eval_env=env_eval,
        n_eval_episodes=cfg.eval_episodes,
        eval_freq=cfg.eval_freq,
        writer=writer,
        verbose=0,
        entropy_samples=cfg.entropy_samples,
    )
    model = make_model(cfg)
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[logging_callback],
    )


def run_trials(
    base_dir: Path, cfg: Namespace, name_props: List[str], num_trials: int
) -> Iterator[Any]:
    name = "_".join(str(getattr(cfg, prop)) for prop in name_props)
    log_dir = base_dir / name
    return (delayed(do_run)(log_dir, cfg, i) for i in range(num_trials))


def run_experiment(name: str, num_trials: int, n_jobs: int) -> None:
    exper_dir = Path("runs") / name
    if exper_dir.exists():
        raise ValueError()

    jobs: List[Tuple] = []
    for env_lsize in range(4, 8):
        cfg = Namespace(**vars(_cfg))
        cfg.action_scale = 2 ** (env_lsize - 4)
        cfg.env_lsize = env_lsize
        jobs.extend(run_trials(exper_dir, cfg, ["env_lsize"], num_trials))
    if len(jobs) == 1 or n_jobs == 1:
        for j in jobs:
            j[0](*j[1], **j[2])
    else:
        Parallel(n_jobs=n_jobs)(j for j in tqdm(jobs))


def patch_old_configs(cfg: Namespace) -> Namespace:
    if not hasattr(cfg, "n_proc_alg"):
        cfg.n_proc_alg = 1
    return cfg


def eval_episode(policy, fe, env, discretize=False) -> Tuple[int, List]:
    obs = env.reset()
    done = False
    steps = 0
    bns = []
    if discretize:
        policy.features_extractor.bottleneck = partial(
            torch.nn.functional.gumbel_softmax, tau=1e-20
        )
    while not done:
        obs_tensor = torch.Tensor(obs)
        with torch.no_grad():
            policy_out = policy(obs_tensor)
            act = policy_out[0].numpy()
            # act, _ = model.predict(obs, state=None, deterministic=True)
            # act = policy_out[0].numpy()
            # act = policy(obs_tensor)[0].numpy()
            bn = fe.forward_bottleneck(obs_tensor).numpy()
        bns.append(bn)
        obs, _, done, _ = env.step(act)
        steps += 1
    return steps, bns


def collect_metrics(path: Path, discretize) -> pd.DataFrame:
    with (path / "config.pkl").open("rb") as fo:
        cfg = pkl.load(fo)
    cfg = patch_old_configs(cfg)
    env = E.Scalable(is_eval=True, **make_env_kwargs(cfg))
    model = make_model(cfg)
    # env = model.env
    # model.load(path / "best.zip")
    # policy = model.policy.cpu()
    policy = model.policy
    policy.load_state_dict(torch.load(path / "best.pt"))
    # features_extractor = model.policy.features_extractor.cpu()
    features_extractor = policy.features_extractor.cpu()
    bottleneck_values = []
    steps_values = []
    for ep in range(cfg_test.n_test_episodes):
        lens, bns = eval_episode(policy, features_extractor, env, discretize)
        steps_values.append(lens)
        bottleneck_values.extend(bns)
    entropies = util.calc_entropies(np.stack(bottleneck_values))
    contents = {
        "steps": np.mean(steps_values),
        **entropies,
        "discretize": discretize,
        **vars(cfg),
    }
    return pd.DataFrame({k: [v] for k, v in contents.items()})


def expand_paths(path_like: Union[str, Path]) -> List[Path]:
    root = Path(path_like)
    if not root.is_dir():
        return []
    contents = {x for x in root.iterdir()}
    names = {x.name for x in contents}
    paths = []
    if len({"best.pt", "config.pkl"} & names) == 2:
        paths.append(root)
    paths.extend(x for c in contents for x in expand_paths(c))
    return paths


def aggregate_results(path_strs: List[str], out_name: str, n_jobs: int) -> None:
    paths = [x for p in path_strs for x in expand_paths(p)]
    jobs = [
        delayed(collect_metrics)(p, discretize=d) for d in (True, False) for p in paths
    ]
    results = Parallel(n_jobs=n_jobs)(x for x in tqdm(jobs))
    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_name, index=False)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("targets", type=str, nargs="+")
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--out_name", "-o", type=str, default="out.csv")
    parser.add_argument("-j", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.command == "test":
        aggregate_results(args.targets, args.out_name, args.j)
    elif args.command == "run":
        run_experiment(args.targets[0], args.num_trials, args.j)
    else:
        raise ValueError(f"Command '{args.command}' not recognized.")


if __name__ == "__main__":
    main()
