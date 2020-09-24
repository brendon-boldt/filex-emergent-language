from typing import Tuple, Any, Dict

import gym  # type: ignore
from gym import spaces
import numpy as np  # type: ignore

IMAGE_RES = 256
EPSILON = 1e-5

StepResult = Tuple[Any, float, bool, Dict]


class Simple(gym.Env):
    def __init__(self):
        super(Simple, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.state = -1.0

    def step(self, action: int) -> StepResult:
        if (action == 0 and self.state < 0) or (action == 1 and self.state > 0):
            reward = 1.0
        else:
            reward = 0.0
        return np.array([self.state]), reward, True, {}

    def reset(self) -> np.ndarray:
        self.state = 1.0 if np.random.default_rng().random() > 0.5 else -1.0
        return np.array([self.state])


class Discrete(gym.Env):
    def __init__(self, grid_size=2):
        super(Discrete, self).__init__()
        self.grid_size = grid_size
        # turn left, turn right, move forward, stop
        self.action_space = spaces.Discrete(4)
        os_low = np.array([[-0.0, -0.0], [-1.0, -1.0], [-0.0, -0.0]])
        # os_low = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        self.observation_space = spaces.Box(low=os_low, high=1.0)

        self.max_steps = 40
        self.num_steps = 0
        self.location = np.zeros((2,), dtype=np.uint8)
        self.goal_location = np.zeros((2,), dtype=np.uint8)
        self.orientation: int = 0
        self.stop = False

    def _take_action(self, action: int) -> None:
        if action == 0:
            # Move forward
            is_x = not bool(self.orientation % 2)
            pm = 1 if self.orientation < 2 else -1
            delta = (is_x * pm, (not is_x) * pm)
            self.location = (self.location + delta).clip(0, self.grid_size - 1)
        elif action == 1:
            # Turn left
            self.orientation = (self.orientation + 1) % 4
        elif action == 2:
            # Turn right
            self.orientation = (self.orientation + 3) % 4
        elif action == 3:
            # Stop
            self.stop = True
        else:
            raise ValueError("Unrecognized action {action}")

    def _get_observation(self) -> StepResult:
        angle = self.orientation * np.pi / 2
        observation = np.array(
            [
                self.location / self.grid_size,
                [np.cos(angle), np.sin(angle)],
                self.goal_location / self.grid_size,
            ]
        )
        diff = (self.location - self.goal_location) / self.grid_size
        # observation = np.array(
        #     [
        #         diff,
        #         [np.cos(angle), np.sin(angle)],
        #     ]
        # )
        reward = 0.0
        if self.stop:
            # reward = 1.0 if (self.location == self.goal_location).all() else -2.0
            # dist = np.sqrt(((observation[0] - observation[2]) ** 2).sum())
            # reward = 1. - dist
            if np.sqrt((diff ** 2).sum()) < 1e-3:
                reward += 100.0
            else:
                reward += 0.0
        else:
            reward += -0.05
        info: Dict[str, Any] = {}
        return observation, reward, self.stop, info

    def step(self, action: int) -> StepResult:
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        self.num_steps += 1
        # If the max has been reached, force the agent to stop
        action = action if self.num_steps < self.max_steps else 3
        self._take_action(action)
        return self._get_observation()

    def reset(self) -> np.ndarray:
        self.location = np.random.default_rng().integers(0, self.grid_size, (2,))
        self.orientation = np.random.default_rng().integers(0, 4)
        self.goal_location = np.random.default_rng().integers(0, self.grid_size, (2,))
        self.stop = False
        self.num_steps = 0
        return self._get_observation()[0]

    def render(self, mode="human", close=False):
        loc_x, loc_y = self.location / self.grid_size
        print(f"{'location':20s} {loc_x:.3f} {loc_y:.3f}")
        ori_x = np.cos(self.orientation * np.pi / 2)
        ori_y = np.sin(self.orientation * np.pi / 2)
        print(f"{'orientation':20s} {ori_x:+.3f} {ori_y:+.3f}")
        gloc_x, gloc_y = self.goal_location / self.grid_size
        print(f"{'goal':20s} {gloc_x:.3f} {gloc_y:.3f}")


class DiscreteAbsolute(gym.Env):
    def __init__(self, grid_size=2):
        super(DiscreteAbsolute, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        channels = 2
        obs_shape = (self.grid_size, self.grid_size, channels)
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

        self.max_steps = 40
        self.num_steps = 0
        self.location = np.zeros((2,), dtype=np.uint8)
        self.goal_location = np.zeros((2,), dtype=np.uint8)
        self.orientation: int = 0
        self.stop = False

    def _take_action(self, action: int) -> None:
        if action == 0:
            # Move forward
            is_x = not bool(self.orientation % 2)
            pm = 1 if self.orientation < 2 else -1
            delta = (is_x * pm, (not is_x) * pm)
            self.location = (self.location + delta).clip(0, self.grid_size - 1)
        elif action == 1:
            # Turn left
            self.orientation = (self.orientation + 1) % 4
        elif action == 2:
            # Turn right
            self.orientation = (self.orientation + 3) % 4
        elif action == 3:
            # Stop
            self.stop = True
        else:
            raise ValueError("Unrecognized action {action}")

    def _get_observation(self) -> StepResult:
        _observation = np.zeros(self.observation_space.shape)
        # Channel 0: agent, 1: goal
        _observation[tuple(self.location) + (0,)] = 1.0 - EPSILON
        next_location = np.array((0, 1) if self.orientation % 2 else (0, 1))
        next_location *= 1 if self.orientation < 2 else -1
        next_location += self.location
        if (next_location >= 0).all() and (next_location < self.grid_size).all():
            _observation[next_location[0], next_location[1], 0] = 0.5
        # Differentiate from zero padding
        _observation[..., 1] = 0.1
        _observation[tuple(np.concatenate([self.goal_location, [1]]))] = 1.0 - EPSILON
        observation = (_observation * 255).astype(np.uint8)

        diff = (self.location - self.goal_location) / self.grid_size
        reward = 0.0
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

    def step(self, action: int) -> StepResult:
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        self.num_steps += 1
        # If the max has been reached, force the agent to stop
        action = action if self.num_steps < self.max_steps else 3
        self._take_action(action)
        return self._get_observation()

    def reset(self) -> np.ndarray:
        self.location = np.random.default_rng().integers(0, self.grid_size, (2,))
        self.orientation = np.random.default_rng().integers(0, 4)
        self.goal_location = np.random.default_rng().integers(0, self.grid_size, (2,))
        self.stop = False
        self.num_steps = 0
        return self._get_observation()[0]

    def render(self, mode="human", close=False):
        loc_x, loc_y = self.location / self.grid_size
        print(f"{'location':20s} {loc_x:.3f} {loc_y:.3f}")
        ori_x = np.cos(self.orientation * np.pi / 2)
        ori_y = np.sin(self.orientation * np.pi / 2)
        print(f"{'orientation':20s} {ori_x:+.3f} {ori_y:+.3f}")
        gloc_x, gloc_y = self.goal_location / self.grid_size
        print(f"{'goal':20s} {gloc_x:.3f} {gloc_y:.3f}")


class Continuous(gym.Env):
    def __init__(self):
        super(Continuous, self).__init__()

        self.max_speed = 1 / 50
        self.max_turn = 1 / 50

        # Action space: go, turn, end episode (if > 0.5)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,))
        # self.observation_space = spaces.Dict(
        #     goal_location=spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        #     location=spaces.Box(low=0.0, high=1.0, shape=(2,)),
        #     orientation=spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        # )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3, 2))
        self.max_steps = 150
        self.reset()

    def reset(self) -> StepResult:
        self.orientation = np.random.default_rng().random() * 2 * np.pi
        self.location = np.random.default_rng().random(2) * (1 - EPSILON)
        self.goal_location = np.random.default_rng().random(2)
        self.stop = False
        self.num_steps = 0
        return self._get_observation()

    def _take_action(self, action: np.ndarray) -> None:
        # Action contents are [move vs. turn, left vs. right, end_episode]
        if action[2] > 0.5:
            self.stop = True
        else:
            direction = 1.0 if action[1] < 0.5 else -1.0
            ori_delta = (1.0 - action[0]) * self.max_turn * direction
            self.orientation = np.fmod(self.orientation + ori_delta, 2 * np.pi)
            loc_delta = (
                action[0]
                * self.max_speed
                * np.array([np.cos(self.orientation), np.sin(self.orientation)])
            )
            self.location = (self.location + loc_delta).clip(0.0, 1.0 - EPSILON)

    def _get_observation(self) -> StepResult:
        observation = np.array([
            self.location,
            [np.cos(self.orientation), np.sin(self.orientation)],
            self.goal_location
            ])
        info: Dict[str, Any] = {}
        reward = 0.
        if self.stop:
            dist = np.sqrt(((self.location - self.goal_location)**2).sum())
            if dist < 0.1:
                reward = (1. - dist) * 100.
        else:
            reward -= 0.01
        return observation, reward, self.stop, info

    def step(self, action: np.ndarray):
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            # Force stop
            action[2] = 0.9
        self._take_action(action)
        if self.stop:
            raise Exception("Cannot step a stopped environment.")
        return self._get_observation()

    def render(self, mode="human", close=False):
        print(f"{'location':20s} {self.location[0]:.3f} {self.location[1]:.3f}")
        ori_x = np.cos(self.orientation)
        ori_y = np.sin(self.orientation)
        print(f"{'orientation':20s} {ori_x:+.3f} {ori_y:+.3f}")
        gloc_x, gloc_y = self.goal_location
        print(f"{'goal':20s} {self.goal_location[0]:.3f} {self.goal_location[1]:.3f}")


def d_random_walk():
    env = Discrete()
    env.reset()
    env.render()
    for i in range(10):
        # action = np.random.default_rng().integers(0, 3)
        action = 0
        env.step(action)
        print(action)
        env.render()
        print()


def c_random_walk():
    env = Continuous()
    env.reset()
    env.render()
    for i in range(100):
        # action = np.random.default_rng().integers(0, 3)
        action = np.array([1.0, 0.5])
        env.step(action)
        if not i % 10:
            env.render()
            print()


if __name__ == "__main__":
    # TODO Add done action
    # c_random_walk()
    d_random_walk()
