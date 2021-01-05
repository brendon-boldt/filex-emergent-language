from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    for env_lsize in range(4, 9):
        yield {
            "obs_type": "vector",
            "env_lsize": env_lsize,
            "action_scale": 2 ** (env_lsize - 4),
        }
