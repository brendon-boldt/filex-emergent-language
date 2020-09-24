import sys

from stable_baselines3 import DQN, PPO  # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore
from stable_baselines3.dqn import CnnPolicy  # type: ignore

from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore
from stable_baselines3.common.cmd_util import make_vec_env  # type: ignore
from stable_baselines3.common.utils import set_random_seed  # type: ignore

import env as E
from nn import BasicCnn

def make_env(env_constructor, rank, seed=0):
    def _init():
        env = env_constructor()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

N_PROC = 8

""" Scratch
{DQN, A2C, PPO} * {CNN, MLP} * {direct, pixel}

"net_arch": [64] * 3,
grid_size = 4x4

random - 6
DQN CNN pixel (200k, 200k) - 11
DQN MLP pixel (200k, 200k) - 14
DQN MLP direct (200k, 200k) - 6

"""


if __name__ == "__main__":
    alg = DQN
    policy_kwargs = {
        # "features_extractor_class": BasicCnn,
        # "net_arch": [64] * 3,
        "net_arch": [100] * 2,
    }
    # env_lam = lambda: E.Discrete(grid_size=4)
    env_lam = lambda: E.Discrete(grid_size=3)
    # env_lam = lambda: E.DiscreteAbsolute(grid_size=4)
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        # env = SubprocVecEnv([make_env(env_lam, i) for i in range(N_PROC)])
        env = env_lam()

        learning_starts = 300_000
        policy_steps = 200_000
        model = alg(
            # "CnnPolicy",
            "MlpPolicy",
            env,
            learning_starts=learning_starts,
            learning_rate=1e-3,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
        model.learn(total_timesteps=int(policy_steps + learning_starts))
        model.save("dqn_discrete")
        # We can't use a vectorized env for eval
        env = env_lam()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
        print(mean_reward, std_reward)

        del model
        exit()


    model = alg.load("dqn_discrete")
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
