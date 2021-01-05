from typing import Dict, Iterator


def generate_configs() -> Iterator[Dict]:
    for bottleneck_temperature in 0.1, 0.5, 2, 10:
        yield {
            "bottleneck_temperature": bottleneck_temperature,
        }
