from pathlib import Path
from typing import Dict, Any

configs: Dict[str, Dict[str, Any]] = {
    "corr_temp": {
        "path": "results/temperature",
        "type": "correlation",
        "ind_var": "bottleneck_temperature_log",
        "dep_var": "entropy",
        "groups": ["rs_multiplier"],
    },
    "corr_lr": {
        "path": "results/learning_rate",
        "type": "correlation",
        "ind_var": "learning_rate_log",
        "dep_var": "entropy",
        "groups": ["rs_multiplier"],
    },
    "corr_sparsity": {
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
