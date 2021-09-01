from pathlib import Path
from typing import Dict, Any

configs: Dict[str, Dict[str, Any]] = {
    "quick_test": {
        "path": "results/quick_test",
        "type": "correlation",
        "ind_var": "sparsity_log",
        "dep_var": "entropy",
    },
    "temperature": {
        "path": "results/temperature",
        "type": "correlation",
        "ind_var": "bottleneck_temperature_log",
        "dep_var": "entropy",
        "groups": ["sparsity"],
    },
    "learning_rate": {
        "path": "results/learning_rate",
        "type": "correlation",
        "ind_var": "learning_rate_log",
        "dep_var": "entropy",
        "groups": ["sparsity"],
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
        "groups": ["sparsity"],
    },
    "world_radius": {
        "path": "results/world_radius",
        "type": "correlation",
        "ind_var": "world_radius_log",
        "dep_var": "entropy",
        "groups": ["sparsity"],
    },
    "sparsity": {
        "path": "results/sparsity",
        "type": "correlation",
        "ind_var": "sparsity_log",
        "dep_var": "entropy",
        "groups": ["bottleneck_size"],
    },
    "debug": {
        "path": "results/debug",
        "type": "correlation",
        "ind_var": "sparsity_log",
        "dep_var": "entropy",
        "groups": ["bottleneck_size"],
    },
    "nav_to_edges_qual": {
        "path": "results/nav_to_edges",
        "type": "snowflake",
        "groups": ["sparsity", "biased_reward_shaping"],
    },
    "goal_radius": {
        "path": "results/goal_radius",
        "type": "correlation",
        "ind_var": "goal_radius_log",
        "dep_var": "entropy",
        "drop_unsuccessful": True,
        "groups": ["sparsity"],
    },
    "buffer_size": {
        "path": "results/buffer_size",
        "type": "correlation",
        "ind_var": "n_steps_log",
        "dep_var": "entropy",
        "groups": ["sparsity"],
    },
    "train_steps": {
        "path": "results/train_steps",
        "type": "correlation",
        "ind_var": "total_timesteps_log",
        "dep_var": "entropy",
        "groups": ["sparsity"],
    },
}

for k in configs.keys():
    # Auto populate the "name" field based on the key
    configs[k]["name"] = k
    configs[k]["path"] = Path(configs[k]["path"])
