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
    env_class=E.Virtual,
    env_shape="circle",  # square, circle
    bottleneck="gsm",
    bottleneck_temperature=1.0,
    bottleneck_hard=False,
    # reward_structure="proximity",  # constant, none, proximity, constant-only
    policy_net_arch=[0x40] * 0,  # default: [0x40] * 2,
    pre_arch=[0x20, 0x20],
    post_arch=[0x20],
    policy_activation="tanh",
    # obs_type="direction",  # vector, direction
    eval_freq=20_000,
    total_timesteps=500_000,
    # max_step_scale=4.5,  # default: 2.5
    eval_steps=500 * 12,  # 12 is approx the average ep len of a converged model
    fe_out_size=0x10,
    fe_out_ratio=4,
    device="cpu",
    alg=PPO,
    n_steps=0x400,  # Was 0x80
    batch_size=0x100,  # Was 0x100
    learning_rate=3e-4,  # default: 3-e4
    # single_step=False,
    save_all_checkpoints=False,
    init_model_path=None,
    # Virtual args
    reward_structure="cosine",
    obs_type="direction",
    single_step=False,
    goal_radius=1.0,
    world_radius=16.0,
    max_step_scale=3.0,
    variant=None,
    entropy_coef=0.0,
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
        "variant": cfg.variant,
        "goal_radius": cfg.goal_radius,
        "world_radius": cfg.world_radius,
        "env_shape": cfg.env_shape,
        "obs_type": cfg.obs_type,
        "reward_structure": cfg.reward_structure,
        "single_step": cfg.single_step,
        "max_step_scale": cfg.max_step_scale,
    }


def make_policy_kwargs(cfg: Namespace) -> gym.Env:
    return {
        "features_extractor_class": nn.BottleneckPolicy,
        "features_extractor_kwargs": {
            "out_size": cfg.fe_out_size,
            "ratio": cfg.fe_out_ratio,
            "bottleneck": cfg.bottleneck,
            "bottleneck_hard": cfg.bottleneck_hard,
            "pre_arch": cfg.pre_arch,
            "post_arch": cfg.post_arch,
            "temp": cfg.bottleneck_temperature,
            "act": cfg.policy_activation,
        },
        "net_arch": cfg.policy_net_arch,
    }


def make_model(cfg: Namespace) -> Any:
    env_kwargs = make_env_kwargs(cfg)
    env = cfg.env_class(is_eval=False, **env_kwargs)
    policy_kwargs = make_policy_kwargs(cfg)
    alg_kwargs = {
        "n_steps": cfg.n_steps,
        "batch_size": cfg.batch_size,
        "policy_kwargs": policy_kwargs,
        "verbose": 0,
        "learning_rate": cfg.learning_rate,
        "device": cfg.device,
        "ent_coef": cfg.entropy_coef,
    }
    if cfg.alg != PPO:
        del alg_kwargs["n_steps"]
        # del alg_kwargs["batch_size"]
        # del alg_kwargs["learning_rate"]
        # alg_kwargs["n_episodes_rollout"] = 100
    model = cfg.alg(
        "MlpPolicy",
        env,
        **alg_kwargs,
    )
    if cfg.init_model_path is not None:
        try:
            model.policy.load_state_dict(torch.load(cfg.init_model_path))
        except:
            return None
    return model


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
    env_kwargs = make_env_kwargs(cfg)
    env_eval = DummyVecEnv([lambda: cfg.env_class(is_eval=True, **env_kwargs)])
    logging_callback = LoggingCallback(
        eval_env=env_eval,
        n_eval_episodes=cfg.eval_steps,
        eval_freq=cfg.eval_freq,
        writer=writer,
        verbose=0,
        save_all_checkpoints=cfg.save_all_checkpoints,
    )
    model = make_model(cfg)
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
    bn_size = next(policy.features_extractor.post_net.modules())[0].in_features
    for i in range(bn_size):
        x = torch.zeros(bn_size)
        x[i] = 1.0
        with torch.no_grad():
            x = policy.features_extractor.post_net(x)
            x = policy.action_net(x)
        _data.append(x.numpy())
    return np.array(_data)


def is_border(m, i, j) -> bool:
    if i % 2 and m[(i - 1) // 2, j // 2] != m[(i + 1) // 2, j // 2]:
        return True
    elif j % 2 and m[i // 2, (j - 1) // 2] != m[i // 2, (j + 1) // 2]:
        return True
    return False


def get_lexicon_map(features_extractor: torch.nn.Module) -> None:
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
                m[i, j] = features_extractor.pre_net(inp).argmax().item()  # type: ignore
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
    discretize: bool,
) -> Optional[Tuple[pd.DataFrame, List[List]]]:
    with (model_path.parent / "config.pkl").open("rb") as fo:
        cfg = pkl.load(fo)
    cfg = patch_old_configs(cfg)
    env_kwargs = {**make_env_kwargs(cfg), "is_eval": True, "single_step": False}
    env = cfg.env_class(**env_kwargs)
    model = make_model(cfg)
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
    features_extractor = policy.features_extractor.cpu()
    bottleneck_values = []
    steps_values = []
    successes = 0.0
    trajs: List[List] = []
    for ep in range(cfg_test.n_test_episodes):
        ep_len, bns, success, traj = util.eval_episode(
            policy, features_extractor, env, discretize
        )
        successes += success
        steps_values.append(ep_len)
        bottleneck_values.extend(bns)
        trajs.extend(traj)
    np_bn_values = np.stack(bottleneck_values)
    entropies = util.get_metrics(np_bn_values)
    sample_id = str(uuid.uuid4())
    contents = {
        "path": str(model_path),
        "uuid": sample_id,
        "steps": np.mean(steps_values),
        "success_rate": successes / cfg_test.n_test_episodes,
        **entropies,
        "discretize": discretize,
        "usages": np_bn_values.mean(0).tolist(),
        "vectors": vectors.tolist(),
        **vars(cfg),
    }
    return pd.DataFrame({k: [v] for k, v in contents.items()}), trajs


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
) -> None:
    paths = [x for p in path_strs for x in expand_paths(p, progression, target_ts)]
    jobs = [
        delayed(collect_metrics)(p, out_dir, discretize=d)
        # for d in (True, False)
        for d in (True,)
        for p in paths
    ]
    results = [
        r for r in Parallel(n_jobs=n_jobs)(x for x in tqdm(jobs)) if r is not None
    ]
    df_contents = [r[0] for r in results]
    df = pd.concat(df_contents, ignore_index=True)
    df.to_csv(out_dir / "data.csv", index=False)
    trajectories_dir = out_dir / "trajectories"
    if not trajectories_dir.exists():
        trajectories_dir.mkdir()
    for vals, trajs in results:
        select = lambda x: np.array([t[x] for t in trajs])
        np.savez(
            trajectories_dir / (vals["uuid"][0] + ".npz"),
            t=select(0),
            s=select(1),
            a=select(2),
            r=select(3),
            s_next=select(4),
            done=select(5),
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("targets", type=str, nargs="*")
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--out_dir", "-o", type=str, default=".")
    parser.add_argument("--progression", action="store_true")
    parser.add_argument("--target_ts", type=int, default=None)
    parser.add_argument("-j", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    args.out_dir = Path(args.out_dir)

    if args.command == "test":
        aggregate_results(
            args.targets, args.out_dir, args.j, args.progression, args.target_ts
        )
    elif args.command == "run":
        run_experiments(args.targets, args.num_trials, args.j)
    else:
        raise ValueError(f"Command '{args.command}' not recognized.")


if __name__ == "__main__":
    main()
