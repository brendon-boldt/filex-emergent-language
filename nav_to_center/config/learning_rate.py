from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    n = 400
    hi = -1
    lo = -4
    for rs_multiplier in [1e-4, 1]:
        for i in range(n):
            x = 10 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "rs_multiplier": rs_multiplier,
                "learning_rate": x,
            }
