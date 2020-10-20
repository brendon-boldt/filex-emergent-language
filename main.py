import sys

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3  # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore
from stable_baselines3.dqn import CnnPolicy  # type: ignore
from stable_baselines3.common import callbacks  # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv  # type: ignore
from stable_baselines3.common.cmd_util import make_vec_env  # type: ignore
from stable_baselines3.common.utils import set_random_seed  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore

import env as E
import nn


def make_env(env_constructor, rank, seed=0):
    def _init():
        env = env_constructor()
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def full_test():
    N_PROC = 1

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
    env_lam = lambda: VecTransposeImage(
        DummyVecEnv([lambda: E.Orientationless(grid_size=3)])
    )
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


def test_1d() -> None:
    lsize = 5
    size = 2 ** lsize
    cnn_out_size = lsize
    cnn_ratio = 1
    percpetion_model = nn.Cnn1D(cnn_out_size, lsize, cnn_ratio, 2)
    policy_model = nn.Perceptron([cnn_out_size * cnn_ratio, 10, 1])
    model = torch.nn.Sequential(percpetion_model, policy_model)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng()

    data_len = 100
    data_y = []
    data_x = []
    for _ in range(data_len):
        loc = rng.integers(0, size)
        gloc = rng.integers(0, size)
        while loc != gloc:
            gloc = rng.integers(0, size)
        data_x.append(torch.zeros((2, size)))
        # TODO Figure out correct dtype
        data_x[-1][0, loc] = 1.0
        data_x[-1][1, gloc] = 1.0
        data_y.append(torch.tensor(1.0 if loc < gloc else -1.0))
    for epoch in range(10):
        running_loss = 0
        for i, idx in enumerate(rng.choice(len(data_x), len(data_x))):
            # y_pred = policy_model(percpetion_model(data_x[idx]))
            y_pred = model(data_x[idx].unsqueeze(0)).squeeze(1).squeeze(1)
            loss = loss_fn(data_y[idx].unsqueeze(0), y_pred)
            running_loss += (loss - running_loss) / (i + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch:03d}  loss: {running_loss:.3f}")

if __name__ == '__main__':
    test_1d()
