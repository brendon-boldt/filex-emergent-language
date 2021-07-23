from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    n = 400
    hi = 3
    lo = 10
    for rs_multiplier in [1e-4, 1]:
        prev_x = None
        for i in range(n):
            x = int(2 ** (lo + (hi - lo) * i / (n - 1)))
            # Because of squashing with int, skip over duplicates
            if x == prev_x:
                continue
            prev_x = x
            yield {
                "pre_arch": [0x20, x],
                "rs_multiplier": rs_multiplier,
            }
