from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    for size in 3, 4, 5, 6, 7, 12, 16, 32, 128:
        yield {"pre_arch": [0x10, size]}
