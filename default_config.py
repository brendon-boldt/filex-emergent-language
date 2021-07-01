import argparse

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3  # type: ignore

import env as E

cfg = argparse.Namespace(
    env_class=E.Virtual,
    bottleneck="gsm",
    bottleneck_temperature=1.5,
    bottleneck_hard=False,
    # policy_net_arch=[0x40] * 0,  # default: [0x40] * 2,
    pre_arch=[0x20, 0x20],
    post_arch=[0x20],
    policy_activation="tanh",
    # obs_type="direction",  # vector, direction
    eval_freq=20_000,
    total_timesteps=500_000,
    eval_steps=1_000,
    device="cpu",
    alg=PPO,
    n_steps=0x400,  # Was 0x80
    batch_size=0x100,
    learning_rate=3e-3,
    save_all_checkpoints=False,
    init_model_path=None,
    # Virtual args
    # TODO Rename
    reward_shape_type="cosine",
    base_reward_type="every-step",
    obs_type="vector",
    goal_radius=1.0,
    world_radius=9.0,
    # TODO Rename this since it is confusing
    max_step_scale=3.0,
    variant=None,
    entropy_coef=0.0,
    gamma=0.9,
    half_life=float("inf"),
    reward_scale=0.1,
    rs_multiplier=0.0,
)
