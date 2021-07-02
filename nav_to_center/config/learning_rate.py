from typing import Dict, Iterator
import env  # type: ignore


def generate_configs() -> Iterator[Dict]:
    base = {
        "total_timesteps": 100_000,
        "eval_freq": 5_000,
        "reward_scale": 1.0,
    }
    n = 40
    hi = -1
    lo = -4
    for i in range(n):
        x = 10 ** (lo + (hi - lo) * i / (n - 1))
        yield {
            "learning_rate": x,
            **base,
        }
