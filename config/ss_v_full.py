from typing import Dict, Iterator
import env  # type: ignore


def generate_configs() -> Iterator[Dict]:
    base = {
        "env_class": env.Virtual,
        "save_all_checkpoints": True,
    }
    # for single_step in False, True:
    #     yield {"single_step": single_step, **base}
    # yield {"variant": "triangle-init", "single_step": True, **base}
    # yield {"reward_structure": "cosine-only", **base}
    yield {"reward_structure": "constant", **base}
