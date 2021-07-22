from typing import Tuple, List, Dict, Any, Callable
from functools import partial
from argparse import Namespace

import numpy as np  # type: ignore
import torch
import gym  # type: ignore
from stable_baselines3 import PPO  # type: ignore

from . import nn
from . import env as E


def _xlx(x: float) -> float:
    if x == 0.0:
        return 0.0
    else:
        return -x * np.log2(x)


xlx = np.vectorize(_xlx)


def get_entropy(o: np.ndarray) -> np.ndarray:
    """Calculate bits of entropy."""
    return xlx(np.eye(o.shape[-1])[o.argmax(-1)].mean(0)).sum(0)


def eval_episode(policy, fe, env, discretize) -> Tuple[int, List, float]:
    obs = env.get_observation()
    done = False
    steps = 0
    bns = []
    original_bottlenck = policy.mlp_extractor.bottleneck
    if discretize:
        policy.mlp_extractor.bottleneck = partial(
            torch.nn.functional.gumbel_softmax,
            hard=True,
            tau=1e-20,
        )
    total_reward = 0.0
    while not done:
        obs_tensor = torch.Tensor(obs)
        with torch.no_grad():
            policy_out = policy(obs_tensor.unsqueeze(0), deterministic=True)
            act = policy_out[0][0].numpy()
            bn = fe.forward_bottleneck(obs_tensor).numpy()
        bns.append(bn)
        prev_loc = env.location.copy()
        obs, reward, done, info = env.step(act)
        total_reward += reward
        steps += 1
    policy.mlp_extractor.bottleneck = original_bottlenck
    total_reward = float(info["at_goal"])
    return steps, bns, total_reward


def make_env_kwargs(cfg: Namespace) -> Dict:
    return {
        "rs_multiplier": cfg.rs_multiplier,
        "goal_radius": cfg.goal_radius,
        "world_radius": cfg.world_radius,
        "max_step_scale": cfg.max_step_scale,
    }


def _make_policy_kwargs(cfg: Namespace) -> gym.Env:
    return {
        "net_arch": {
            "bottleneck_hard": cfg.bottleneck_hard,
            "pre_arch": cfg.pre_arch,
            "post_arch": cfg.post_arch,
            "temp": cfg.bottleneck_temperature,
            "act": cfg.policy_activation,
        },
    }


def make_model(cfg: Namespace) -> Any:
    env_kwargs = make_env_kwargs(cfg)
    env = E.NavToCenter(is_eval=False, **env_kwargs)
    policy_kwargs = _make_policy_kwargs(cfg)
    alg_kwargs = {
        "n_steps": cfg.n_steps,
        "batch_size": cfg.batch_size,
        "policy_kwargs": policy_kwargs,
        "verbose": 0,
        "learning_rate": cfg.learning_rate,
        "device": cfg.device,
        "ent_coef": cfg.entropy_coef,
        "gamma": cfg.gamma,
    }
    if cfg.alg != PPO:
        del alg_kwargs["n_steps"]
    model = cfg.alg(
        nn.MixedPolicy,
        env,
        **alg_kwargs,
    )
    if cfg.init_model_path is not None:
        try:
            model.policy.load_state_dict(torch.load(cfg.init_model_path))
        except:
            return None
    return model
