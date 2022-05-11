from pathlib import Path
from typing import Dict, Any

configs: Dict[str, Dict[str, Any]] = {
    "quick_test": {
        "path": "results/quick_test",
        "type": "correlation",
        "ind_var": "learning_rate_log",
        "dep_var": "entropy",
    },
    "learning_rate_perf": {
        "path": "results/learning_rate",
        "ind_var": "learning_rate_log",
        "dep_var": "steps",
    },
    "learning_rate_align": {
        "path": "results/learning_rate",
        "type": "align",
        "ind_var": "learning_rate_log",
        "model_data_path": "model-rust/output/alpha.out",
    },
    "learning_rate": {
        "ind_var": "learning_rate_log",
    },
    "lexicon_size_perf": {
        "path": "results/lexicon_size",
        "ind_var": "bottleneck_size_log",
        "dep_var": "steps",
    },
    "lexicon_size": {
        "ind_var": "bottleneck_size_log",
    },
    "buffer_size_nd": {
        "ind_var": "n_steps_log",
    },
    "temperature": {
        "ind_var": "bottleneck_temperature_log",
    },
    "buffer_size": {
        "ind_var": "n_steps_log",
    },
    "buffer_size_perf": {
        "path": "results/buffer_size",
        "ind_var": "n_steps_log",
        "dep_var": "steps",
    },
    "train_steps_perf": {
        "path": "results/train_steps",
        "ind_var": "total_timesteps_log",
        "dep_var": "steps",
    },
    "train_steps": {
        "ind_var": "total_timesteps_log",
    },
    **{
        name: {
            "ind_var": var,
            "drop_unsuccessful": False,
        }
        for name, var in [
            ("rc_time_steps", "total_timesteps_log"),
            ("rc_buffer_size", "n_steps_log"),
            ("rc_lexicon_size", "bottleneck_size_log"),
            ("rc_learning_rate", "learning_rate_log"),
            ("rc_temperature", "bottleneck_temperature_log"),
            ("sg_time_steps", "total_timesteps_log"),
            ("sg_temperature", "bottleneck_temperature_log"),
            ("sg_buffer_size", "n_steps_log"),
            ("sg_lexicon_size", "bottleneck_size_log"),
            ("sg_learning_rate", "learning_rate_log"),
        ]
    },
}

for k, cfg in configs.items():
    # Auto populate the "name" field based on the key
    cfg["name"] = k
    cfg["path"] = Path(cfg.get("path", f"results/{k}"))
    cfg["type"] = cfg.get("type", "correlation") 
    cfg["dep_var"] = cfg.get("dep_var", "entropy") 
