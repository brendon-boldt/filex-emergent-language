from typing import Dict, Iterator
import env  # type: ignore


def generate_configs() -> Iterator[Dict]:
    base = {
        "total_timesteps": 40_000,
        "eval_freq": 5_000,
        "reward_shape_type":"cosine",
        "reward_scale": 1.0,
    }
    n = 100
    hi = 0
    lo = -4
    for i in range(n):
        x = 10 ** (lo + (hi - lo) * i / (n-1))
        yield {
            # 'base_reward_type': x,
            'rs_multiplier': x,
            **base,
        }
