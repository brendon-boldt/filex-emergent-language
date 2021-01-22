from typing import Tuple, List
from functools import partial

import numpy as np  # type: ignore
import torch

from typing import Dict


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


def eval_episode(policy, fe, env, discretize=False) -> Tuple[int, List, float, List]:
    obs = env.reset()
    done = False
    steps = 0
    bns = []
    original_bottlenck = policy.features_extractor.bottleneck
    if discretize:
        policy.features_extractor.bottleneck = partial(
            torch.nn.functional.gumbel_softmax, hard=True
        )
    total_reward = 0.0
    traj: List[List] = []
    while not done:
        obs_tensor = torch.Tensor(obs)
        with torch.no_grad():
            policy_out = policy(obs_tensor, deterministic=True)
            if env.discrete_action:
                act = np.int64(policy_out[0].numpy())
            else:
                if type(policy_out) == tuple:
                    act = policy_out[0].numpy()
                else:
                    act = policy_out.numpy()
            bn = fe.forward_bottleneck(obs_tensor).numpy()
        bns.append(bn)
        prev_obs = obs
        obs, reward, done, info = env.step(act)
        traj.append([steps, prev_obs, act, reward, obs, info['at_goal']])
        total_reward += reward
        steps += 1
    policy.features_extractor.bottleneck = original_bottlenck
    if hasattr(env, "use_reward") and not env.use_reward:
        pass
    else:
        total_reward = float(info['at_goal'])
    return steps, bns, total_reward, traj
