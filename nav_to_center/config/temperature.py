from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    base = {
        "total_timesteps": 100_000,
        "eval_freq": 5_000,
    }
    n = 40
    hi = 1
    lo = -1
    for i in range(n):
        x = 2 ** (lo + (hi - lo) * i / (n - 1))
        yield {
            "bottleneck_temperature": x,
            **base,
        }
