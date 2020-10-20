import sys

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3  # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore
from stable_baselines3.dqn import CnnPolicy  # type: ignore
from stable_baselines3.common import callbacks  # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv  # type: ignore
from stable_baselines3.common.cmd_util import make_vec_env  # type: ignore
from stable_baselines3.common.utils import set_random_seed  # type: ignore

import env as E
import nn


def make_env(env_constructor, rank, seed=0):
    def _init():
        env = env_constructor()
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


""" Scratch
{DQN, A2C, PPO} * {CNN, MLP} * {direct, pixel}

"net_arch": [64] * 3,
grid_size = 4x4

random - 6
DQN CNN pixel (200k, 200k) - 11
DQN MLP pixel (200k, 200k) - 14
DQN MLP direct (200k, 200k) - 6


On direct 4x4, A2C gets 50% after 1.3m @ 8/16 processes with 32k/16k udpates
"""

N_PROC = 1

if __name__ == "__main__":
    # alg = DQN
    # alg = A2C
    # alg = PPO
    alg = SAC
    policy_kwargs = {
        # "features_extractor_class": nn.BasicCnn,
        "features_extractor_class": nn.SimpleCnn,
        "net_arch": [0x20] * 2,
    }
    # env_lam = lambda: E.Discrete(grid_size=4)
    # env_lam = lambda: VecTransposeImage(DummyVecEnv([lambda: E.DiscreteAbsolute(grid_size=4)]))
    env_lam = lambda: VecTransposeImage(DummyVecEnv([lambda: E.Orientationless(grid_size=3)]))
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        # env = SubprocVecEnv([make_env(env_lam, i) for i in range(N_PROC)])
        env_eval = env_lam()
        env = env_eval

        learning_starts = 200_000
        policy_steps = 300_000
        model = alg(
            "CnnPolicy",
            # "MlpPolicy",
            env,
            # learning_starts=learning_starts,
            # n_steps=10,
            # learning_rate=1e-4,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        model.learn(
            total_timesteps=int(policy_steps + learning_starts),
            # log_interval=5_000,
            callback=[
                callbacks.EvalCallback(
                    eval_env=env_eval, n_eval_episodes=1000, eval_freq=1_000
                )
            ],
        )
        model.save("model-save")
        # We can't use a vectorized env for eval
        mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=1000)
        print(mean_reward, std_reward)

        del model
        exit()

    model = alg.load("model-save")
    env = env_lam()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(mean_reward, std_reward)
    for i in range(0):
        obs = env.reset()
        total_reward = 0.0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            print(f"action: {action}")
            obs, rewards, dones, info = env.step(action)
            total_reward += rewards
            # print(obs)
            if env.stop:
                # print(obs)
                print(env.location)
                print(env.goal_location)
                print(total_reward)
                break
