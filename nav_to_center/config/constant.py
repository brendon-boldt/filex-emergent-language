from typing import Dict, Iterator
import env  # type: ignore


def generate_configs() -> Iterator[Dict]:
    base = {
        "learning_rate": 3e-3,
        "total_timesteps": 200_000,
    }
    for bns in 0x10, 0x20:
        for rs in ["cosine-only", "euclidean"]:
            for rsm_exp in [-float("inf"), -3, -2, -1, 0]:
                yield {
                    "reward_structure": rs,
                    "pre_arch": [0x20, bns],
                    "rs_multiplier": 10 ** rsm_exp,
                    **base,
                }
