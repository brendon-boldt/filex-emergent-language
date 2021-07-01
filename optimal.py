from typing import Any, Callable, List, Tuple
import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np  # type: ignore
from joblib import Parallel, delayed  # type: ignore
import pandas as pd  # type: ignore

from default_config import cfg as _cfg
from run import make_env_kwargs
import env

rng = np.random.default_rng()

Policy = Callable[[np.ndarray], np.ndarray]


def make_policy(n: int) -> Policy:
    # offset = rng.random() * np.pi * 2 / n
    offset = 0.1
    angles = offset + np.arange(n) * np.pi * 2 / n
    acts = np.array([np.cos(angles), np.sin(angles)]).T

    def f(obs: np.ndarray) -> np.ndarray:
        # return obs * 100
        return acts[(acts @ obs).argmax()].copy()

    return f


def test_policy(env: Any, n: int, samples: int = 10_000) -> Tuple:
    p = make_policy(n)
    stepss: List[int] = []
    g_rad = env.goal_radius / env.world_radius
    lo = int(np.ceil(samples * g_rad ** 2))
    hi = int(np.ceil(samples / (1 - g_rad ** 2)))
    for i in range(lo, hi):
        env.reset()
        obs = env.fib_disc_init(i, hi)
        steps = 0
        done = False
        while not done:
            steps += 1
            act = p(obs)
            obs, _, done, _ = env.step(act)
        stepss.append(steps)
    # print(f"{n:>3}: {np.mean(stepss):.2f}")
    # print(f"{n:>3}, {np.log2(n):.3f}, {np.mean(stepss):f}")
    return n, -np.mean(stepss)


def run_optimal_agents() -> None:
    env_kwargs = {
        **make_env_kwargs(_cfg),
        "is_eval": True,
        "single_step": False,
        "world_radius": 9.0,
        "obs_type": "vector",
    }
    new_env = lambda: env.NavToCenter(**env_kwargs)
    results = Parallel(n_jobs=20)(
        delayed(test_policy)(new_env(), i, 10_000) for i in range(3, 33)
    )
    for r in results:
        print(f"{r[0]},{r[1]}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.command == "run":
        run_optimal_agents()
    # elif args.command == "spline":
    #     generate_spline_data(args.target)
    else:
        raise ValueError(f"Command '{args.command}' not recognized.")


if __name__ == "__main__":
    main()
