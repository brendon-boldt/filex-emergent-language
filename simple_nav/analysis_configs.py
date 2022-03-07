from pathlib import Path
from typing import Dict, Any

configs: Dict[str, Dict[str, Any]] = {
    "quick_test": {
        "path": "results/quick_test",
        "type": "correlation",
        "ind_var": "learning_rate_log",
        "dep_var": "entropy",
    },
    "learning_rate": {
        "path": "results/learning_rate",
        "type": "correlation",
        "ind_var": "learning_rate_log",
        "dep_var": "entropy",
    },
    "lexicon_size": {
        "path": "results/lexicon_size",
        "type": "correlation",
        "ind_var": "bottleneck_size_log",
        "dep_var": "entropy",
    },
    "buffer_size": {
        "path": "results/buffer_size",
        "type": "correlation",
        "ind_var": "n_steps_log",
        "dep_var": "entropy",
    },
    "train_steps": {
        "path": "results/train_steps",
        "type": "correlation",
        "ind_var": "total_timesteps_log",
        "dep_var": "entropy",
    },
}

for k in configs.keys():
    # Auto populate the "name" field based on the key
    configs[k]["name"] = k
    configs[k]["path"] = Path(configs[k]["path"])
