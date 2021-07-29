from pathlib import Path
from typing import Dict, Any

configs: Dict[str, Dict[str, Any]] = {
    "quick_test": {
        "path": "results/quick_test",
        "type": "correlation",
        "ind_var": "rs_multiplier_log",
        "dep_var": "entropy",
    },
    "temperature": {
        "path": "results/temperature",
        "type": "correlation",
        "ind_var": "bottleneck_temperature_log",
        "dep_var": "entropy",
        "groups": ["rs_multiplier"],
    },
    "learning_rate": {
        "path": "results/learning_rate",
        "type": "correlation",
        "ind_var": "learning_rate_log",
        "dep_var": "entropy",
        "groups": ["rs_multiplier"],
    },
    "low_learning_rate": {
        "path": "results/low_learning_rate",
        "type": "correlation",
        "ind_var": "learning_rate_log",
        "dep_var": "entropy",
    },
    "bottleneck_size": {
        "path": "results/bottleneck_size",
        "type": "correlation",
        "ind_var": "bottleneck_size_log",
        "dep_var": "entropy",
        "groups": ["rs_multiplier"],
    },
    "sparsity": {
        "path": "results/sparsity",
        "type": "correlation",
        "ind_var": "rs_multiplier_log",
        "dep_var": "entropy",
    },
}

for k in configs.keys():
    # Auto populate the "name" field based on the key
    configs[k]["name"] = k
    configs[k]["path"] = Path(configs[k]["path"])
