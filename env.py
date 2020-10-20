from typing import Tuple, Any, Dict

import gym  # type: ignore
from gym import spaces
import numpy as np  # type: ignore

IMAGE_RES = 256
EPSILON = 1e-5

StepResult = Tuple[Any, float, bool, Dict]

class Env2D(gym.Env):
    def __init__(self, grid_size):
        super(self.__class__, self).__init__()
        self.grid_size = grid_size
        # self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Box(low=-1., high=1., shape=(3,))
        channels = 2
        obs_shape = (self.grid_size, self.grid_size, channels)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        self.max_steps = self.grid_size * 3
        self.num_steps = 0
        self.location = np.zeros((2,), dtype=np.uint8)
        self.goal_location = np.zeros((2,), dtype=np.uint8)
        self.stop = False

    def _take_action(self, action: np.ndarray) -> None:
        # TODO Incorporate variable resolution
        if action[2] > 0.0:
            self.stop = True
            return
        move = action[:2]
        gran = 4
        tau = np.pi * 2
        binned_angle = np.round(gran * np.arctan2(*reversed(move)) / tau) * tau / gran
        delta = np.array([np.cos(binned_angle), np.sin(binned_angle)])
        self.location = (self.location + delta).clip(0, self.grid_size - 1).astype(np.int32)

    def _get_observation(self) -> StepResult:
        _observation = np.zeros(self.observation_space.shape)
        # Channel 0: agent, 1: agent next, 2: goal
        # x and y are reversed when indexing; maybe I should change this?
        _observation[self.location[1], self.location[0], 0] = 1.0 - EPSILON
        # Differentiate from zero padding
        # _observation[..., 2] = 0.1
        _observation[self.goal_location[1], self.goal_location[0], 1] = 1.0 - EPSILON
        observation = (_observation * 255).astype(np.uint8)

        diff = (self.location - self.goal_location) / self.grid_size
        reward = 0.0
        if self.num_steps > 0 and np.sqrt((diff ** 2).sum()) < 1e-3:
            self.stop = True
        if self.stop:
            # reward = 1.0 if (self.location == self.goal_location).all() else -2.0
            if np.sqrt((diff ** 2).sum()) < 1e-3:
                reward += 100.0
            else:
                reward += 0.0
        else:
            reward += -0.05
        info: Dict[str, Any] = {}
        return observation, reward, self.stop, info

    def step(self, action: np.ndarray) -> StepResult:
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        self.num_steps += 1
        # If the max has been reached, force the agent to stop
        action = action if self.num_steps < self.max_steps else np.array([0.0, 0.0, 1.0])
        self._take_action(action)
        return self._get_observation()

    def reset(self) -> np.ndarray:
        self.location = np.random.default_rng().integers(0, self.grid_size, (2,))
        self.goal_location = np.random.default_rng().integers(0, self.grid_size, (2,))
        self.stop = False
        self.num_steps = 0
        return self._get_observation()[0]


def test_disc_abs():
    env = DiscreteAbsolute(grid_size=3)
    obs = env.reset()
    print(obs.transpose(-1, 0, 1)[:2])
    stop = False
    i = 0
    while not stop:
        action = i % 2
        i = (i + 1) % 2
        obs, _, stop, _ = env.step(action)
        print(action, env.orientation)
        print(obs.transpose(-1, 0, 1)[:2])

class Env1D(gym.Env):
    def __init__(self, *, lsize: int, obs_lscale: int, action_scale: int, max_steps: int = None):
        super(self.__class__, self).__init__()

        # natural number, >= 0
        self.lsize = lsize

        # how many loci (grid squares) is the world across
        self.size = int(2 ** self.lsize)

        self.location = 0
        self.goal_location = self.size - 1

        # what granularity does the agent observe
        self.obs_lscale = obs_lscale
        self.obs_dim = int(2 ** obs_lscale)

        # what the network outputs
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))

        # the largest number of loci the agent can move in one step
        self.action_scale = action_scale

        channels = 2
        obs_shape = (self.grid_size, channels)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        self.max_steps = int(self.size * 1.5)

        # For type purposes
        self.num_steps = 0
        self.location = 0
        self.goal_location = 0
        self.stop = False

    def _take_action(self, action: np.ndarray) -> None:
        self.location = np.clip(int(np.round(self.location + action[0] * self.action_scale)), 0, self.size - 1)

    def _get_observation(self) -> StepResult:
        _observation = np.zeros(self.observation_space.shape)
        _observation[self.location, 0] = 1.0 - EPSILON
        # _observation[..., 1] = 0.1
        _observation[self.goal_location, 1] = 1.0 - EPSILON
        observation = (_observation * 255).astype(np.uint8)

        diff = (self.location - self.goal_location) / self.size
        reward = 0.0
        if self.num_steps > 0 and np.sqrt(diff ** 2) < 1e-3:
            self.stop = True
        if self.stop:
            if np.sqrt(diff ** 2) < 1e-3:
                reward += 100.0
            else:
                reward += 0.0
        else:
            reward += -0.05
        info: Dict[str, Any] = {}
        return observation, reward, self.stop, info

    def step(self, action: np.ndarray) -> StepResult:
        # TODO Keep making changes below
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        self.num_steps += 1
        # If the max has been reached, force the agent to stop
        action = action if self.num_steps < self.max_steps else np.array([0.0, 0.0, 1.0])
        self._take_action(action)
        return self._get_observation()

    def reset(self) -> np.ndarray:
        self.location = np.random.default_rng().integers(0, self.grid_size, (2,))
        self.goal_location = np.random.default_rng().integers(0, self.grid_size, (2,))
        self.stop = False
        self.num_steps = 0
        return self._get_observation()[0]

if __name__ == "__main__":
    test_disc_abs()
