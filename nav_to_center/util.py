from typing import Tuple, List, Dict, Any
from functools import partial
from argparse import Namespace

import numpy as np  # type: ignore
import torch
import gym  # type: ignore
from stable_baselines3 import PPO  # type: ignore

from . import nn
from . import env


def _xlx(x: float) -> float:
    if x == 0.0:
        return 0.0
    else:
        return -x * np.log2(x)


xlx = np.vectorize(_xlx)


def get_metrics(o: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate entropies based on network outputs."""
    return {
        "argmax": xlx(np.eye(o.shape[-1])[o.argmax(-1)].mean(0)).sum(0),
        "fractional": xlx(o.mean(0)).sum(0),
        "linf": o.max(-1).mean(0),
        "individual": xlx(o).sum(-1).mean(0),
    }


def eval_episode(policy, fe, env, discretize) -> Tuple[int, List, float, List]:
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
    traj: List[List] = []
    while not done:
        obs_tensor = torch.Tensor(obs)
        with torch.no_grad():
            policy_out = policy(obs_tensor.unsqueeze(0), deterministic=True)
            act = policy_out[0][0].numpy()
            bn = fe.forward_bottleneck(obs_tensor).numpy()
        bns.append(bn)
        prev_loc = env.location.copy()
        obs, reward, done, info = env.step(act)
        traj.append(
            [steps, prev_loc, act, reward, env.location.copy(), info["at_goal"]]
        )
        total_reward += reward
        steps += 1
    policy.mlp_extractor.bottleneck = original_bottlenck
    if hasattr(env, "use_reward") and not env.use_reward:
        pass
    else:
        total_reward = float(info["at_goal"])
    return steps, bns, total_reward, traj


def make_env(env_constructor, rank, seed=0):
    def _init():
        env = env_constructor()
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def make_env_kwargs(cfg: Namespace) -> Dict:
    return {
        "rs_multiplier": cfg.rs_multiplier,
        "goal_radius": cfg.goal_radius,
        "world_radius": cfg.world_radius,
        "max_step_scale": cfg.max_step_scale,
    }


def make_policy_kwargs(cfg: Namespace) -> gym.Env:
    return {
        # "features_extractor_class": nn.BottleneckPolicy,
        # "features_extractor_kwargs": {
        #     "out_size": cfg.fe_out_size,
        #     "ratio": cfg.fe_out_ratio,
        #     "bottleneck": cfg.bottleneck,
        #     "bottleneck_hard": cfg.bottleneck_hard,
        #     "pre_arch": cfg.pre_arch,
        #     "post_arch": cfg.post_arch,
        #     "temp": cfg.bottleneck_temperature,
        #     "act": cfg.policy_activation,
        # },
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
    env = env.NavToCenter(is_eval=False, **env_kwargs)
    policy_kwargs = make_policy_kwargs(cfg)
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
        # del alg_kwargs["batch_size"]
        # del alg_kwargs["learning_rate"]
        # alg_kwargs["n_episodes_rollout"] = 100
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
