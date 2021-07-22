from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    n = 40
    hi = 1
    lo = -1
    for sparsity in [1e-4, 1]:
        for i in range(n):
            x = 2 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "rs_multiplier": sparsity,
                "bottleneck_temperature": x,
            }
