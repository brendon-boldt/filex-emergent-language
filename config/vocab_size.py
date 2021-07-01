from typing import Dict, Iterator
import env  # type: ignore


def generate_configs() -> Iterator[Dict]:
    base = {
        "total_timesteps": 100_000,
        "eval_freq": 5_000,
        "reward_shape_type":"cosine",
        "reward_scale": 1.0,
    }
    n = 120
    hi = 4 
    lo = 12
    for i in range(n):
        x = 2 ** (lo + (hi - lo) * i / (n-1))
        yield {
            'pre_arch': [0x20, int(x)],
            **base,
        }
