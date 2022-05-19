from typing import Tuple, List, Dict, Any, Iterable
from functools import partial
from argparse import Namespace

from stable_baselines3.common.utils import obs_as_tensor
import numpy as np  # type: ignore
import torch  # type: ignore
import gym  # type: ignore
from stable_baselines3 import PPO, A2C  # type: ignore

from . import nn
from . import env as E


def xlx(x: np.ndarray) -> np.ndarray:
    return -x * np.log2(x + 1e-20)


def get_entropy(o: np.ndarray) -> np.ndarray:
    """Calculate bits of entropy."""
    return xlx(np.eye(o.shape[-1])[o.argmax(-1)].mean(0)).sum(0)


def eval_episode(policy, fe, env, discretize) -> Dict[str, Any]:
    obs = env.reset()
    done = False
    steps = 0
    bns = []
    logitss = []
    bns_s = []
    original_bottlenck = policy.mlp_extractor.bottleneck
    if discretize:
        policy.mlp_extractor.bottleneck = partial(
            torch.nn.functional.gumbel_softmax,
            hard=True,
            tau=1e-20,
        )
    total_reward = 0.0
    while not done:
        obs_tensor = obs_as_tensor(obs, 'cpu')  # type: ignore
        with torch.no_grad():
            policy_out = policy(obs_tensor, deterministic=True)
            act = policy_out[0][0].numpy()
            bn_results = fe._logits, fe._bn_activations[0]
            bna_soft = original_bottlenck(bn_results[0])
        logitss.append(bn_results[0].numpy())
        bns.append(bn_results[1].numpy())
        bns_s.append(bna_soft.numpy())
        obs, reward, done, info = env.step(np.expand_dims(act, 0))
        total_reward += reward[0]
        steps += 1
    policy.mlp_extractor.bottleneck = original_bottlenck
    total_reward = float(info[0].get("success", 0))
    return {
        "steps": steps,
        "bn_activations": bns,
        "bn_activations_soft": bns_s,
        "bn_logits": logitss,
        "total_reward": total_reward,
    }


def make_env_kwargs(cfg: Namespace) -> Dict:
    return {
        "sparsity": cfg.sparsity,
        "goal_radius": cfg.goal_radius,
        "world_radius": cfg.world_radius,
        "max_step_scale": cfg.max_step_scale,
        "biased_reward_shaping": cfg.biased_reward_shaping,
        "n_dims": cfg.n_dims,
        "n_opts": cfg.n_opts,
    }


def _make_policy_kwargs(cfg: Namespace) -> gym.Env:
    return {
        "net_arch": {
            "bottleneck_hard": cfg.bottleneck_hard,
            "pre_bottleneck_arch": cfg.pre_bottleneck_arch,
            "post_bottleneck_arch": cfg.post_bottleneck_arch,
            "temp": cfg.bottleneck_temperature,
            "act": cfg.policy_activation,
            "signal_game": cfg.environment == E.Signal,
            "n_opts": cfg.n_opts,
        },
    }


def make_model(cfg: Namespace) -> Any:
    env_kwargs = make_env_kwargs(cfg)
    env = cfg.environment(is_eval=False, **env_kwargs)
    policy_kwargs = _make_policy_kwargs(cfg)
    alg_kwargs = {
        "n_steps": cfg.n_steps,
        "batch_size": cfg.batch_size,
        "policy_kwargs": policy_kwargs,
        "verbose": 0,
        "learning_rate": cfg.learning_rate,
        "device": "cpu",
        "gamma": cfg.gamma,
    }
    if cfg.rl_algorithm == A2C:
        del alg_kwargs["batch_size"]
    policy = nn.MultiInputBottleneckPolicy if cfg.environment == E.Signal else nn.BottleneckPolicy
    model = cfg.rl_algorithm(
        policy,
        env,
        **alg_kwargs,
    )
    if cfg.init_model_path is not None:
        try:
            model.policy.load_state_dict(torch.load(cfg.init_model_path))
        except Exception:
            return None
    return model

def log_range(low: float, high: float, steps: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(low), np.log(high), steps))
