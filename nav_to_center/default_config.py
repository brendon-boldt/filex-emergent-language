import argparse

from stable_baselines3 import PPO  # type: ignore

cfg = argparse.Namespace(
    bottleneck_temperature=1.5,
    bottleneck_hard=False,
    pre_arch=[0x20, 0x20],
    post_arch=[0x20],
    policy_activation="tanh",
    eval_freq=5_000,
    total_timesteps=100_000,
    eval_steps=1_000,
    device="cpu",
    alg=PPO,
    n_steps=0x400,  # Was 0x80
    batch_size=0x100,
    learning_rate=3e-3,
    init_model_path=None,
    goal_radius=1.0,
    world_radius=9.0,
    # TODO Rename this since it is confusing
    max_step_scale=3.0,
    entropy_coef=0.0,
    gamma=0.9,
    rs_multiplier=0.0,
)
