import sys
from typing import Any, Tuple, List, Callable, Iterator, Union, Dict, Optional
import argparse
from argparse import Namespace
from pathlib import Path
import pickle as pkl
from itertools import chain
import importlib
import shutil
import uuid

import gym  # type: ignore
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3  # type: ignore
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
from scipy.interpolate import CubicSpline  # type: ignore
from stable_baselines3.common.callbacks import EvalCallback
from PIL import Image  # type: ignore

import nn
import env
from callback import LoggingCallback
import util
from default_config import cfg as _cfg


def do_run(base_dir: Path, cfg: argparse.Namespace, idx: int) -> None:
    log_dir = base_dir / f"run-{idx}"
    if (log_dir / "completed").exists():
        return
    elif log_dir.exists():
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    with (log_dir / "config.txt").open("w") as text_fo:
        text_fo.write(str(cfg))
    with (log_dir / "config.pkl").open("wb") as binary_fo:
        pkl.dump(cfg, binary_fo)
    env_kwargs = util.make_env_kwargs(cfg)
    env_eval = DummyVecEnv([lambda: env.NavToCenter(is_eval=True, **env_kwargs)])
    logging_callback = LoggingCallback(
        eval_env=env_eval,
        n_eval_episodes=cfg.eval_steps,
        eval_freq=cfg.eval_freq,
        writer=writer,
        verbose=0,
        save_all_checkpoints=cfg.save_all_checkpoints,
        cfg=cfg,
    )
    model = util.make_model(cfg)
    if model is None:
        raise ValueError("Could not restore model.")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[logging_callback],
    )
    # Create empty file to show the run is completed in case it gets interrupted
    # halfway through.
    (log_dir / "completed").open("w")


def run_trials(
    base_dir: Path, cfg: Namespace, name_props: List[str], num_trials: int
) -> Iterator[Any]:
    name = "_".join(str(getattr(cfg, prop)).replace("/", ",") for prop in name_props)
    log_dir = base_dir / name
    return (delayed(do_run)(log_dir, cfg, i) for i in range(num_trials))


def run_experiments(
    config_paths: List[str], num_trials: int, n_jobs: int, out_dir=Path("log")
) -> None:
    jobs: List[Tuple] = []
    for config_path in config_paths:
        config_name = config_path.split("/")[-1][:-3]
        out_path = out_dir / config_name
        module_name = config_path.rstrip("/").replace("/", ".")[:-3]
        mod: Any = importlib.import_module(module_name)
        for config in mod.generate_configs():
            final_config = {**vars(_cfg), **config}
            cfg = Namespace(**final_config)
            jobs.extend(run_trials(out_path, cfg, list(config.keys()), num_trials))

    if len(jobs) == 1 or n_jobs == 1:
        for j in jobs:
            j[0](*j[1], **j[2])
    else:
        Parallel(n_jobs=n_jobs)(j for j in tqdm(jobs))


def patch_old_configs(cfg: Namespace) -> Namespace:
    if not hasattr(cfg, "base_reward_type"):
        cfg.reward_scale = "each-step"
    if not hasattr(cfg, "reward_scale"):
        cfg.reward_scale = 0.1
    if not hasattr(cfg, "rs_multiplier"):
        cfg.rs_multiplier = 1.0
    if not hasattr(cfg, "half_life"):
        cfg.half_life = float("inf")
    if not hasattr(cfg, "gamma"):
        cfg.gamma = 0.99
    if not hasattr(cfg, "bottleneck_hard"):
        cfg.bottleneck_hard = False
    if not hasattr(cfg, "entropy_coef"):
        cfg.entropy_coef = 0.0
    if not hasattr(cfg, "variant"):
        cfg.variant = None
    if not hasattr(cfg, "init_model_path"):
        cfg.init_model_path = None
    return cfg


def get_one_hot_vectors(policy: Any) -> np.ndarray:
    _data = []
    bn_size = next(policy.mlp_extractor.post_net.modules())[0].in_features
    for i in range(bn_size):
        x = torch.zeros(bn_size)
        x[i] = 1.0
        with torch.no_grad():
            x = policy.mlp_extractor.post_net(x)
            x = policy.action_net(x)
        _data.append(x.numpy())
    return np.array(_data)


def is_border(m, i, j) -> bool:
    if i % 2 and m[(i - 1) // 2, j // 2] != m[(i + 1) // 2, j // 2]:
        return True
    elif j % 2 and m[i // 2, (j - 1) // 2] != m[i // 2, (j + 1) // 2]:
        return True
    return False


def get_lexicon_map(mlp_extractor: torch.nn.Module) -> None:
    n_divs = 40
    m = np.zeros([n_divs + 1] * 2, dtype=np.int64)
    bound = 1.0
    print()
    print()
    for i in range(n_divs + 1):
        for j in range(n_divs + 1):
            with torch.no_grad():
                inp = torch.tensor(
                    [-bound + 2 * bound * i / n_divs, -bound + 2 * bound * j / n_divs]
                )
                m[i, j] = mlp_extractor.pre_net(inp).argmax().item()  # type: ignore
    for i in range(2 * n_divs - 1):
        for j in range(2 * n_divs - 1):
            c = "  "
            if not i % 2 and not j % 2:
                c = f"{m[i//2,j//2]:>2d}"
            elif i % 2 and j % 2:
                if sum(
                    [
                        is_border(m, i + x, j + y)
                        for x, y in [(-1, 0), (0, -1), (1, 0), (0, 1)]
                    ]
                ):
                    c = "██"
            elif is_border(m, i, j):
                c = "██"
            print(c, end="")
        print()


def collect_metrics(
    model_path: Path,
    out_path: Path,
    eval_steps: int,
) -> Optional[pd.DataFrame]:
    with (model_path.parent / "config.pkl").open("rb") as fo:
        cfg = pkl.load(fo)
    cfg = patch_old_configs(cfg)
    env_kwargs: Dict[str, Any] = {
        **util.make_env_kwargs(cfg),
        "is_eval": True,
        "world_radius": 9.0,
    }
    env = env.NavToCenter(**env_kwargs)
    model = util.make_model(cfg)
    if model is None:
        print(f'Could not restore model "{cfg.init_model_path}"')
        return None
    policy = model.policy
    try:
        policy.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(e)
        return None
    vectors = get_one_hot_vectors(model.policy)
    mlp_extractor = policy.mlp_extractor.cpu()
    bottleneck_values = []
    steps_values = []
    successes = 0.0
    trajs: List[List] = []
    # n_episodes = 0
    # TODO
    n_episodes = eval_steps
    n_steps = 0
    discretize = cfg.bottleneck != "none"

    g_rad = env.goal_radius / env.world_radius
    lo = int(np.ceil(n_episodes * g_rad ** 2))
    hi = int(np.ceil(n_episodes / (1 - g_rad ** 2)))

    # while n_steps < eval_steps:
    for i in range(lo, hi):
        n_episodes += 1
        env.reset()
        env.fib_disc_init(i, hi)
        ep_len, bns, success, traj = util.eval_episode(
            policy, mlp_extractor, env, discretize
        )
        n_steps += ep_len
        successes += success
        steps_values.append(ep_len)
        bottleneck_values.extend(bns)
        trajs.extend(traj)
    np_bn_values = np.stack(bottleneck_values)
    entropies = util.get_metrics(np_bn_values)
    sample_id = str(uuid.uuid4())

    trajectories_dir = out_path / "trajectories"
    if not trajectories_dir.exists():
        trajectories_dir.mkdir()
    select = lambda x: np.array([t[x] for t in trajs])
    np.savez(
        trajectories_dir / (sample_id + ".npz"),
        t=select(0),
        s=select(1),
        a=select(2),
        r=select(3),
        s_next=select(4),
        done=select(5),
    )

    contents = {
        "path": str(model_path),
        "uuid": sample_id,
        "steps": np.mean(steps_values),
        "success_rate": successes / n_episodes,
        **entropies,
        "discretize": discretize,
        "usages": np_bn_values.mean(0).tolist(),
        "vectors": vectors.tolist(),
        **vars(cfg),
    }
    return pd.DataFrame({k: [v] for k, v in contents.items()})


def expand_paths(
    path_like: Union[str, Path], progression: bool, target_ts: Optional[int]
) -> List[Path]:
    root = Path(path_like)
    if not root.is_dir():
        return []
    contents = {x for x in root.iterdir()}
    names = {x.name for x in contents}
    paths = []
    target_fn = "best.pt" if target_ts is None else f"model-{target_ts}.pt"
    if progression:
        new_paths = sorted(
            root.glob("model*.pt"),
            key=lambda x: int(str(x).split("-")[-1].split(".")[0]),
        )
        paths.extend(new_paths)
    elif len({target_fn, "config.pkl"} & names) == 2:
        paths.append(root / target_fn)
    paths.extend(x for c in contents for x in expand_paths(c, progression, target_ts))
    return paths


def aggregate_results(
    path_strs: List[str],
    out_dir: Path,
    n_jobs: int,
    progression: bool,
    target_ts: Optional[int],
    eval_steps: int,
    df_concat_paths: List[Path],
) -> None:
    dfs_to_concat = [pd.read_csv(p) for p in df_concat_paths]
    if not out_dir.exists():
        out_dir.mkdir()
    paths = [x for p in path_strs for x in expand_paths(p, progression, target_ts)]
    jobs = [delayed(collect_metrics)(p, out_dir, eval_steps) for p in paths]
    results = [
        r for r in Parallel(n_jobs=n_jobs)(x for x in tqdm(jobs)) if r is not None
    ]
    df_contents = results
    df = pd.concat(df_contents + dfs_to_concat, ignore_index=True)
    df.to_csv(out_dir / "data.csv", index=False)


def generate_spline_data(path: Path) -> None:
    df = pd.read_csv(path, header=None)
    df[0] = np.log2(df[0])
    cs = CubicSpline(df[0], df[1])
    interps = [[x, cs(x)] for x in np.arange(df[0].min(), df[0].max(), 0.02)]
    df = pd.concat([df, pd.DataFrame(interps)])
    df.to_csv("optimal-interp.csv", header=None)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("targets", type=str, nargs="*")
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--out_dir", "-o", type=str, default=".")
    parser.add_argument("--progression", action="store_true")
    parser.add_argument("--target_ts", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=10_000)
    parser.add_argument("--include_csv", type=str, action="append")
    parser.add_argument("-j", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    args.out_dir = Path(args.out_dir)
    if args.include_csv is None:
        args.include_csv = []

    if args.command == "eval":
        if args.target_ts is None:
            raise ValueError("Please set --target_ts")
        aggregate_results(
            args.targets,
            args.out_dir,
            args.j,
            args.progression,
            args.target_ts,
            args.eval_steps,
            args.include_csv,
        )
    elif args.command == "spline":
        generate_spline_data(args.targets[0])
    elif args.command == "run":
        run_experiments(args.targets, args.num_trials, args.j)
    else:
        raise ValueError(f"Command '{args.command}' not recognized.")


if __name__ == "__main__":
    main()
