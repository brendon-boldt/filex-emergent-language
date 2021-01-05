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
    env_lsize=7,
    action_scale=2 ** 3,
    discrete_action=False,
    bottleneck="gsm",
    bottleneck_temperature=1.0,
    reward_structure="proximity",  # constant, none, proximity, constant-only
    policy_net_arch=[0x40] * 0,  # default: [0x40] * 2,
    pre_arch=[0x10, 0x10],
    post_arch=[0x10],
    policy_activation="tanh",
    action_noise=0.0,
    obs_type="vector",  # vector, direction
    entropy_samples=400,
    eval_freq=20000,
    total_timesteps=5_000_000,
    reward_threshold=0.95,
    max_step_scale=4.5,  # default: 2.5
    eval_episodes=500,
    fe_out_size=0x10,
    fe_out_ratio=4,
    pixel_space=False,
    device="cpu",
    n_proc_alg=1,
    alg=PPO,
    n_steps=0x400,  # Was 0x80
    batch_size=0x100,  # Was 0x100
    learning_rate=3e-4,  # default: 3-e4
    single_step=False,
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
        "obs_type": cfg.obs_type,
        "action_noise": cfg.action_noise,
        "reward_structure": cfg.reward_structure,
        "discrete_action": cfg.discrete_action,
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
            "temp": cfg.bottleneck_temperature,
            "act": cfg.policy_activation,
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
    for els in range(4, 8):
        # for bn_size in (3, 4, 6, 8, 32, 64, 128, 1024):
        cfg = Namespace(**vars(_cfg))
        # cfg.pre_arch = list(cfg.pre_arch)
        # cfg.pre_arch[-1] = bn_size
        cfg.obs_type = "direction"
        cfg.env_lsize = els
        cfg.action_scale = 2 ** (els - 4)
        jobs.extend(run_trials(exper_dir, cfg, ["env_lsize"], num_trials))
    if len(jobs) == 1 or n_jobs == 1:
        for j in jobs:
            j[0](*j[1], **j[2])
    else:
        Parallel(n_jobs=n_jobs)(j for j in tqdm(jobs))


def patch_old_configs(cfg: Namespace) -> Namespace:
    if not hasattr(cfg, "obs_type"):
        cfg.obs_type = "vector"
    if not hasattr(cfg, "policy_activation"):
        cfg.policy_activation = "tanh"
    if not hasattr(cfg, "bottleneck_temperature"):
        cfg.action_noise = 0.0
    if not hasattr(cfg, "bottleneck_temperature"):
        cfg.bottleneck_temperature = 1.0
    if not hasattr(cfg, "reward_structure"):
        cfg.reward_structure = "proximity"
    if not hasattr(cfg, "n_proc_alg"):
        cfg.n_proc_alg = 1
    if not hasattr(cfg, "discrete_action"):
        cfg.discrete_action = False
    return cfg


def eval_episode(policy, fe, env, discretize=False) -> Tuple[int, List, bool]:
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
            if env.discrete_action:
                act = np.int64(policy_out[0].numpy())
            else:
                act = policy_out[0].numpy()
            # act, _ = model.predict(obs, state=None, deterministic=True)
            # act = policy_out[0].numpy()
            # act = policy(obs_tensor)[0].numpy()
            bn = fe.forward_bottleneck(obs_tensor).numpy()
        bns.append(bn)
        obs, _, done, info = env.step(act)
        steps += 1
    return steps, bns, info["at_goal"]


def get_one_hot_vectors(policy: Any) -> np.ndarray:
    _data = []
    bn_size = next(policy.features_extractor.post_net.modules())[0].in_features
    for i in range(bn_size):
        x = torch.zeros(bn_size)
        x[i] = 1.0
        with torch.no_grad():
            x = policy.features_extractor.post_net(x)
            x = policy.action_net(x)
        _data.append(x.numpy())
    return np.array(_data)


def collect_metrics(path: Path, discretize) -> pd.DataFrame:
    with (path / "config.pkl").open("rb") as fo:
        cfg = pkl.load(fo)
    cfg = patch_old_configs(cfg)
    env = E.Scalable(is_eval=True, **make_env_kwargs(cfg))
    model = make_model(cfg)
    policy = model.policy
    policy.load_state_dict(torch.load(path / "best.pt"))
    vectors = get_one_hot_vectors(model.policy)
    features_extractor = policy.features_extractor.cpu()
    bottleneck_values = []
    steps_values = []
    successes = 0
    for ep in range(cfg_test.n_test_episodes):
        lens, bns, success = eval_episode(policy, features_extractor, env, discretize)
        successes += success
        steps_values.append(lens)
        bottleneck_values.extend(bns)
    np_bn_values = np.stack(bottleneck_values)
    entropies = util.get_metrics(np_bn_values)
    contents = {
        "steps": np.mean(steps_values),
        "success_rate": successes / cfg_test.n_test_episodes,
        **entropies,
        "discretize": discretize,
        "usages": np_bn_values.mean(0).tolist(),
        "vectors": vectors.tolist(),
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


def optimality_test() -> None:
    cfg = _cfg
    env_kwargs = make_env_kwargs(cfg)
    env = E.Scalable(is_eval=True, **env_kwargs)
    rng = np.random.default_rng()
    n_epochs = 1000
    for n_divs in [3, 4, 5, 6, 7, 8, 12, 16, 32]:
        stepss = []
        for _ in range(n_epochs):
            sep_angle = 2 * np.pi / n_divs
            base_angle = rng.uniform(sep_angle)
            angles = np.array([base_angle + i * sep_angle for i in range(n_divs)])
            vecs = np.array([[np.cos(a), np.sin(a)] for a in angles])
            obs = env.reset()
            done = False
            steps = 0
            while not done and steps < 100:
                dists = ((obs - vecs) ** 2).sum(-1)
                act = vecs[dists.argsort()[0]]
                obs, _, done, info = env.step(act)
                steps += 1
            stepss.append(steps)
        step_arr = np.array(stepss)
        mean = step_arr.mean()
        stderr = step_arr.std() / np.sqrt(n_epochs)
        print(f"{n_divs:>2d}: {mean:.1f} +- {2*stderr:.2f}")


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
    elif args.command == "optimal":
        optimality_test()
    else:
        raise ValueError(f"Command '{args.command}' not recognized.")


if __name__ == "__main__":
    main()
