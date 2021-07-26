from typing import Dict, Iterator
import env  # type: ignore


def learning_rate() -> Iterator[Dict]:
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


def low_learning_rate() -> Iterator[Dict]:
    base = {
        "total_timesteps": 500_000,
    }
    n = 100
    hi = -3
    lo = -7
    # for rs_multiplier in [1e-4, 1]:
    for rs_multiplier in [1]:
        for i in range(n):
            x = 10 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "rs_multiplier": rs_multiplier,
                "learning_rate": x,
                **base,
            }


def lexicon_size() -> Iterator[Dict]:
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


def sparsity() -> Iterator[Dict]:
    base = {
        "total_timesteps": 40_000,
    }
    n = 1000
    hi = 0
    lo = -4
    for i in range(n):
        x = 10 ** (lo + (hi - lo) * i / (n - 1))
        yield {
            "rs_multiplier": x,
            **base,
        }


def temperature() -> Iterator[Dict]:
    n = 400
    hi = 1
    lo = -1
    for rs_multiplier in [1e-4, 1]:
        for i in range(n):
            x = 2 ** (lo + (hi - lo) * i / (n - 1))
            yield {
                "rs_multiplier": rs_multiplier,
                "bottleneck_temperature": x,
            }
