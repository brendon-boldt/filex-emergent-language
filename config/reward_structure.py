from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    for s in "constant", "none", "constant-only":
        yield {"reward_structure": s}
