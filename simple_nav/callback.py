import warnings
import argparse
from typing import List, Union

import torch  # type: ignore
import gym  # type: ignore
import numpy as np  # type: ignore
from stable_baselines3.common.callbacks import EventCallback  # type: ignore
from stable_baselines3.common.vec_env import (  # type: ignore
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization,
)
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from . import util


class EvalCallback(EventCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        writer: SummaryWriter,
        cfg: argparse.Namespace,
        n_eval_episodes: int = 200,
        eval_freq: int = 5000,
    ) -> None:
        super(self.__class__, self).__init__(None, verbose=0)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.writer = writer
        self.cfg = cfg

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert (
                eval_env.num_envs == 1
            ), "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.log_path = self.writer.log_dir
        self.evaluations_results: List[float] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

    def _on_step(self) -> bool:
        # if not (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
        if not (self.n_calls % self.eval_freq == 0):
            return True
        # Sync training and eval env if there is VecNormalize
        sync_envs_normalization(self.training_env, self.eval_env)

        if self.model is None or self.model.policy is None:
            raise ValueError("Model/policy is None.")
        # env = self.eval_env.envs[0]
        env = self.eval_env
        results = []
        for _ in range(self.n_eval_episodes):
            ep_results = util.eval_episode(
                self.model.policy,
                self.model.policy.mlp_extractor,
                env,
                True,
            )
            results.append(ep_results)

        episode_rewards = [r["total_reward"] for r in results]
        episode_lengths = [r["steps"] for r in results]
        bn_activations = np.concatenate([r["bn_activations"] for r in results])
        bn_activations_soft = np.concatenate([r["bn_activations_soft"] for r in results])

        entropy = util.get_entropy(bn_activations)
        self.writer.add_scalar("entropy", entropy, self.num_timesteps)

        # bas_shape = bn_activations_soft.shape
        # new_shape = (bas_shape[0] // self.cfg.n_steps, self.cfg.n_steps, bas_shape[-1])
        # truncated_len = bas_shape[0] - (bas_shape[0] % self.cfg.n_steps)
        # if truncated_len:
        #     bas_grouped = bn_activations_soft[:truncated_len].reshape(new_shape)
        #     H_updates_batched = util.xlx(bas_grouped.mean(1)).sum(-1).mean(0)
        #     H_updates = util.xlx(bn_activations_soft.mean(0)).sum(0)
        #     self.writer.add_scalar("H_updates_diff", H_updates - H_updates_batched, self.num_timesteps)

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(sum(episode_rewards) / len(episode_rewards))
            self.evaluations_length.append(np.mean(episode_lengths))
            np.savez(
                self.log_path / "data",
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
            )

        mean_reward, _ = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, _ = np.mean(episode_lengths), np.std(episode_lengths)
        self.writer.add_scalar("mean_reward", float(mean_reward), self.num_timesteps)
        self.writer.add_scalar("mean_ep_length", mean_ep_length, self.num_timesteps)
        self.writer.add_scalar("rate", self.num_timesteps, self.num_timesteps)
        torch.save(
            self.model.policy.state_dict(),
            self.log_path / f"model-{self.num_timesteps}.pt",
        )
        return True


class GradCallback(EventCallback):
    def __init__(
        self,
        writer: SummaryWriter,
        eval_freq: int = 10,
    ) -> None:
        super(self.__class__, self).__init__(None, verbose=0)
        self.eval_freq = eval_freq
        self.writer = writer

    def _on_step(self) -> bool:
        return True
        if not (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            return True

        # self.writer.add_scalar("entropy", entropy, self.num_timesteps)

        if self.model is None or self.model.policy is None:
            raise ValueError("Model/policy is None.")
        bn_acts = self.model.policy.mlp_extractor.bn_activations
        # breakpoint()
        self.model.policy.mlp_extractor.bn_activations = []

        # self.writer.add_scalar("entropy", entropy, self.num_timesteps)

        return True
