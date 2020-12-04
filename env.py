from typing import Tuple, Any, Dict

import gym  # type: ignore
from gym import spaces
import numpy as np  # type: ignore

StepResult = Tuple[Any, float, bool, Dict]

rng = np.random.default_rng()


def norm(x):
    return np.sqrt((x ** 2).sum(-1))


class Scalable(gym.Env):
    def __init__(
        self,
        pixel_space: bool,
        *,
        lsize: int,
        obs_lscale: int = None,
        action_scale: int = 1,
        max_step_scale: float = 2.5,
        single_step=False,
        is_eval=False,
    ) -> None:
        super(self.__class__, self).__init__()
        self.pixel_space = pixel_space
        if obs_lscale is None:
            obs_lscale = lsize

        # natural number, >= 0
        self.lsize = lsize

        # how many loci (grid squares) is the world across
        self.size = int(2 ** self.lsize)

        # For typing/clarity purposes
        self.location = np.array([0] * 2)
        self.goal_location = np.array([self.size - 1] * 2)

        # what granularity does the agent observe
        self.obs_lscale = obs_lscale
        self.obs_dim = int(2 ** obs_lscale)

        # what the network outputs
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        # the largest number of loci the agent can move in one step
        self.action_scale = action_scale

        # If true, use a toy environment with one-step episodes
        self.single_step = single_step
        self.is_eval = is_eval

        if self.pixel_space:
            channels = 2
            obs_shape = (*2 * (self.size,), channels)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

        # TODO Should this be affected by action scale?
        self.max_steps = int((self.size / self.action_scale) * max_step_scale)

        # For type purposes
        self.num_steps = 0
        self.stop = False

    def _take_action(self, action: np.ndarray) -> None:
        assert action.shape == (2,)
        if True:
            # Correct way
            action = np.round(action * self.action_scale)
            act_norm = norm(action)
            if act_norm > self.action_scale:
                action /= act_norm
        else:
            act_norm = norm(action)
            if act_norm > 1:
                action /= act_norm
            action = np.round(action * self.action_scale)
        self.location = np.clip(
            # np.trunc(self.location + action * self.action_scale).astype(np.int32),
            # np.round(self.location + action * self.action_scale).astype(np.int32),
            self.location + action.astype(np.int32),
            0,
            self.size - 1,
        )

    def _is_at_goal(self) -> bool:
        diff = (self.location - self.goal_location) / self.size
        dist = (diff ** 2).sum() ** 0.5
        # return dist < 1e-3
        return dist < (self.action_scale / self.size)

    def _get_observation(self, prev_location: np.ndarray) -> StepResult:
        info: Dict[str, Any] = {}
        if self.pixel_space:
            _observation = np.zeros(self.observation_space.shape)
            T = tuple
            _observation[T(self.location)][0] = 1.0
            # _observation[..., 1] = 0.1
            _observation[T(self.goal_location)][1] = 1.0
            observation = (_observation * 255).astype(np.uint8)
        else:
            observation = (self.goal_location - self.location) / self.size

        diff = (self.location - self.goal_location) / self.size
        dist = (diff ** 2).sum() ** 0.5
        prev_diff = (prev_location - self.goal_location) / self.size
        prev_dist = (prev_diff ** 2).sum() ** 0.5

        at_goal = self._is_at_goal()
        if (self.num_steps > 0 and at_goal) or self.num_steps > self.max_steps:
            self.stop = True

        if self.single_step:
            reward = 1.0 if dist < prev_dist else -1.0
            self.stop = self.num_steps > 0
        elif self.is_eval:
            reward = float(self.stop and at_goal)
        else:
            reward = 0.0
            reward += -0.01 if dist < prev_dist else -0.02
            if self.stop:
                if at_goal:
                    reward += 1.0
                else:
                    reward += 0
                    # reward += -10.0
            else:
                pass
                # reward += 1. if dist < prev_dist else -2.
        return observation, reward, self.stop, info

    def step(self, action: np.ndarray) -> StepResult:
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        self.num_steps += 1
        # If the max has been reached, force the agent to stop
        prev_location = self.location.copy()
        self._take_action(action)
        return self._get_observation(prev_location)

    def reset(self) -> np.ndarray:
        self.location = rng.integers(0, self.size, (2,))
        self.goal_location = rng.integers(0, self.size, (2,))
        while self._is_at_goal():
            self.goal_location = rng.integers(0, self.size, (2,))
        self.stop = False
        self.num_steps = 0
        return self._get_observation(self.location)[0]
