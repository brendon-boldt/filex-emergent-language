import sys
from typing import Any, Tuple, List

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


device = "cpu"


def make_2d_bc_policy(env) -> Any:
    lr = 5e-4
    n_epochs = 200
    cnn_ratio = 2
    # ((2 ** s) * (2 ** s)) ** 2
    # 0x4000
    data_len = 2 ** (env.lsize * 4)
    batch_size = 2 ** 9

    cnn_out_size = env.lsize
    percpetion_model = nn.ScalableCnn(2, cnn_out_size, env.lsize, cnn_ratio, 2)
    policy_model = nn.Perceptron([cnn_out_size * cnn_ratio, 0x20, 0x20, 0x20, 2])
    model = torch.nn.Sequential(percpetion_model, policy_model)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng()
    _data_y = []
    _data_x = []
    # TODO Let the environment generate the data.
    for _ in range(data_len):
        loc = rng.integers(0, env.size, 2)
        gloc = rng.integers(0, env.size, 2)
        while (loc == gloc).all():
            gloc = rng.integers(0, env.size, 2)
        _data_x.append(np.zeros((2, env.size, env.size), dtype=np.float32))
        # TODO Figure out correct dtype
        _data_x[-1][0, loc[0], loc[1]] = 1.0
        _data_x[-1][1, gloc[0], gloc[1]] = 1.0
        diff = np.array(gloc - loc, dtype=np.float32)
        # y_gold = diff / (diff ** 2).sum() ** 0.5
        # Discretize angle
        disc_angle = (np.pi / 2) * (
            (np.round(2 * np.arctan2(*(gloc - loc)) / (np.pi)) + 4) % 4
        )
        y_gold = np.array([np.sin(disc_angle), np.cos(disc_angle)])
        _data_y.append(y_gold)

    model.to(device)
    data_x = torch.FloatTensor(_data_x).to(device)
    data_y = torch.FloatTensor(_data_y).to(device)

    for epoch in range(n_epochs):
        running_loss = 0
        for i, idx in enumerate(
            rng.choice(len(data_x), len(data_x)).reshape(-1, batch_size)
        ):
            y_pred = model(data_x[idx])
            loss = loss_fn(data_y[idx], y_pred)
            running_loss += (loss - running_loss) / (i + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch:03d}  loss: {running_loss:.3f}")

    for i in range(0):
        pred = model(data_x[[i]]).detach()[0]
        print(data_x[i])
        print(data_y[i])
        print(pred)
        print(loss_fn(data_y[i], pred))
        print()

    return model


def make_2d_rs_policy(env) -> Any:
    lr = 1e-3
    n_iters = 20_000
    cnn_ratio = 2
    # ((2 ** s) * (2 ** s)) ** 2
    # 0x4000
    data_len = 2 ** (env.lsize * 4)
    batch_size = 2 ** 9

    cnn_out_size = env.lsize
    percpetion_model = nn.ScalableCnn(2, cnn_out_size, env.lsize, cnn_ratio, 2)
    policy_model = nn.Perceptron([cnn_out_size * cnn_ratio, 0x20, 0x20, 0x20, 2])
    model = torch.nn.Sequential(percpetion_model, policy_model)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng()

    model.to(device)

    tick = 400
    dist = lambda x, y: np.sqrt(((x - y) ** 2).sum())
    for it in range(n_iters):
        if it % tick == 0:
            if it != 0:
                st = list(zip(*stats))
                m_reward = np.mean(st[0])
                m_loss = np.mean(st[1])
                print(f"{it}  reward: {m_reward:.3f}  loss: {m_loss:.3f}")
            stats: List[Tuple] = []
        obs = env.reset()
        orig_loc = env.location
        action = model(
            torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        ).squeeze()
        env.step(action.detach().cpu().numpy())
        new_loc = env.location
        if dist(new_loc, env.goal_location) < dist(orig_loc, env.goal_location):
            r = 1.0
        else:
            r = -1.0
        # print(action)
        # print(dist(new_loc, env.goal_location), dist(orig_loc, env.goal_location), r)
        loss = -r + (action - action.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats.append((r, loss.detach().cpu().numpy()))
        # print(f"epoch {epoch:03d}  loss: {running_loss:.3f}")

    return model


def test_2d():
    lsize = 1
    size = 2 ** lsize
    cnn_out_size = lsize
    action_scale = 1
    env = E.Scalable(lsize=lsize, obs_lscale=lsize, action_scale=action_scale)

    train_model = True
    model_path = "model.pt"
    if train_model:
        # model = make_2d_bc_policy(env)
        model = make_2d_rs_policy(env)
        torch.save(model, model_path)
    else:
        model = torch.load(model_path)

    model.eval()
    succs = 0
    n_episodes = 100
    verbose = False
    for i in range(n_episodes):
        done = False
        obs = env.reset()
        if verbose:
            print("goal")
            print(env.goal_location)
        while not done:
            obs = (
                torch.FloatTensor(np.moveaxis(obs, -1, 0)).to(device).unsqueeze(0)
                / 255.0
            )
            # print(np.unravel_index(obs[0, 0].argmax(), obs.shape[2:]))
            if verbose:
                print(env.location)
            with torch.no_grad():
                action = model(obs)[0].cpu().numpy()
            if verbose:
                print(action)
            # print(action)
            obs, reward, done, _ = env.step(action)
        succs += reward > 80
    if verbose:
        print(env.goal_location)
    print(succs / n_episodes)


def test_1d() -> None:
    lsize = 5
    size = 2 ** lsize
    cnn_out_size = lsize
    cnn_ratio = 1
    percpetion_model = nn.ScalableCnn(1, cnn_out_size, lsize, cnn_ratio, 2)
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


if __name__ == "__main__":
    # test_1d()
    test_2d()
