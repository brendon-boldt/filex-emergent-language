import sys

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import env as E

env = E.Discrete(grid_size=8)
# env = E.Simple()
if len(sys.argv) >= 2 and sys.argv[1] == "train":
    learning_starts = 50_000
    policy_steps = 800_000
    model = DQN(
        "MlpPolicy",
        env,
        learning_starts=learning_starts,
        learning_rate=1e-3,
        verbose=0,
        policy_kwargs={"net_arch": [100, 100]},
    )
    model.learn(total_timesteps=int(policy_steps + learning_starts))
    model.save("dqn_discrete")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
    print(mean_reward, std_reward)

    del model


model = DQN.load("dqn_discrete")
if True:
    obs = env.reset()
    total_reward = 0
    for i in range(20):
        action, _states = model.predict(obs, deterministic=True)
        print(f"action: {action}")
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards
        # print(obs)
        if env.stop:
            print(obs)
            print(total_reward)
            break
