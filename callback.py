from typing import Union
import os
import warnings

import torch
import gym  # type: ignore
import numpy as np  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common import base_class
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization,
)
from torch.utils.tensorboard import SummaryWriter

import util


class LoggingCallback(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param deterministic: Whether to render or not the environment during evaluation
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        writer: SummaryWriter,
        rolling_mean_samples=10,
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        entropy_samples: int = 0,
        save_all_checkpoints: bool = False,
    ) -> None:
        super(self.__class__, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.writer = writer
        self.entropy_samples = entropy_samples
        self.save_all_checkpoints = save_all_checkpoints

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert (
                eval_env.num_envs == 1
            ), "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.log_path = self.writer.log_dir
        self.evaluations_results: List[List[np.ndarray]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            if self.model is None or self.model.policy is None:
                raise ValueError("Model/policy is None.")
            episode_rewards = []
            episode_lengths = []
            bn_activations = []
            # n_eval_episodes is the _actually_ the number of steps
            while sum(episode_lengths) < self.n_eval_episodes:
                ep_len, bns, success = util.eval_episode(
                    self.model.policy,
                    self.model.policy.features_extractor,
                    self.eval_env.envs[0],
                    True,
                )
                episode_rewards.append(success)
                episode_lengths.append(ep_len)
                bn_activations.extend(bns)
            bn_activations = np.array(bn_activations)

            entropies = util.get_metrics(bn_activations)
            self.writer.add_scalar(
                "entropy/argmax", entropies["argmax"], self.num_timesteps
            )
            # self.writer.add_scalar(
            #     "entropy/frac", entropies["fractional"], self.num_timesteps
            # )
            # self.writer.add_scalar(
            #     "entropy/indiv", entropies["individual"], self.num_timesteps
            # )
            # self.writer.add_scalar(
            #     "entropy/linf", entropies["linf"], self.num_timesteps
            # )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(
                    sum(episode_rewards) / len(episode_rewards)
                )
                self.evaluations_length.append(np.mean(episode_lengths))
                np.savez(
                    self.log_path / "data",
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward
            best_criterion = (
                -np.mean(self.evaluations_length[-10:])
                if len(self.evaluations_length)
                else -np.inf
            )

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            self.writer.add_scalar(
                "mean_reward", float(mean_reward), self.num_timesteps
            )
            self.writer.add_scalar("mean_ep_length", mean_ep_length, self.num_timesteps)
            self.writer.add_scalar("rate", self.num_timesteps, self.num_timesteps)

            if self.entropy_samples > 0 and self.model and self.model.policy:
                _outps = []
                for _ in range(self.entropy_samples):
                    obs = torch.FloatTensor(self.eval_env.reset()).to(
                        self.model.policy.device
                    )
                    _outps.append(
                        self.model.policy.features_extractor.forward_bottleneck(obs)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                outps = np.array(_outps).squeeze(1)

            if self.save_all_checkpoints:
                torch.save(
                    self.model.policy.state_dict(),
                    self.log_path / f"model-{self.num_timesteps}.pt",
                )
            if best_criterion > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                assert self.model
                assert self.model.policy
                self.model.save(
                    self.log_path
                    / "best.zip"
                    # os.path.join(self.best_model_save_path, "best_model")
                )
                torch.save(self.model.policy.state_dict(), self.log_path / "best.pt")
                self.best_mean_reward = best_criterion
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        # Causes a mypy error
        # if self.callback:
        #     self.callback.update_locals(locals_)
        pass
