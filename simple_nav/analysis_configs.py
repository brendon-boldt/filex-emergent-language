from pathlib import Path
from typing import Dict, Any

configs: Dict[str, Dict[str, Any]] = {
    "quick_test": {
        "ind_var": "learning_rate_log",
        "drop_unsuccessful": False,
    },
    **{
        f"{env}_{exp}": {
            "ind_var": ind_var,
            "drop_unsuccessful": False,
        } for env in ["nodyn", "recon", "sig", "nav"]
        for exp, ind_var in [
            ("timesteps", "total_timesteps_log"),
            ("temperature", "bottleneck_temperature_log"),
            ("lexicon_size", "bottleneck_size_log"),
            ("learning_rate", "learning_rate_log"),
            ("buffer_size", "n_steps_log"),
            ]
    },
}

for k, cfg in configs.items():
    # Auto populate the "name" field based on the key
    cfg["name"] = k
    cfg["path"] = Path(cfg.get("path", f"results/{k}"))
    cfg["type"] = cfg.get("type", "correlation") 
    cfg["dep_var"] = cfg.get("dep_var", "entropy") 
