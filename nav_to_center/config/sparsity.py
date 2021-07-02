from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    base = {
        "total_timesteps": 40_000,
        "eval_freq": 5_000,
    }
    n = 100
    hi = 0
    lo = -4
    for i in range(n):
        x = 10 ** (lo + (hi - lo) * i / (n - 1))
        yield {
            "rs_multiplier": x,
            **base,
        }
