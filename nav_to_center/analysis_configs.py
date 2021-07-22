from pathlib import Path
from typing import Dict, Any

configs: Dict[str, Dict[str, Any]] = {
    "corr_temp": {
        "path": "results/temperature",
        "type": "correlation",
        "ind_var": "bottleneck_temperature_log",
        "dep_var": "entropy",
        "groups": ["rs_multiplier"],
    }
}

for k in configs.keys():
    # Auto populate the "name" field based on the key
    configs[k]["name"] = k
    configs[k]["path"] = Path(configs[k]["path"])
