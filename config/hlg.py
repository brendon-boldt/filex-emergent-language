from typing import Dict, Iterator
import env  # type: ignore


def generate_configs() -> Iterator[Dict]:
    base = {
        "total_timesteps": 200_000,
        'rs_multiplier': 0.01,
    }
    # for bns in 0x10, 0x20:
    for rs in ['cosine-only', 'euclidean']:
        yield base
        for gamma in [0.8, 0.6, 0.4, 0.2, 0]:
            yield {
                "reward_structure": rs,
                "gamma": gamma,
                'note': 'gamma',
                **base,
            }
        for hl in [8.0, 4.0, 2.0, 1.0]:
            yield {
                "reward_structure": rs,
                "half_life": hl,
                'note': 'hl',
                **base,
            }
