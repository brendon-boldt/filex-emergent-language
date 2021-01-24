from typing import Tuple, Any, Dict, Union, cast

import gym  # type: ignore
from gym import spaces
import numpy as np  # type: ignore

StepResult = Tuple[Any, float, bool, Dict]

rng = np.random.default_rng()


def get_norm(x):
    return np.sqrt((x ** 2).sum(-1))


def cosine_distance(x, y) -> float:
    return (x * y).sum() / max(get_norm(x) * get_norm(y), 1e-5)


class Supervised(gym.Env):
    # TODO Randomize
    angle_offset = np.pi / 4.0
    # TODO Parameterize
    n_actions = 8

    def __init__(
        self,
        *args,
        is_eval=False,
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__()
        if self.angle_offset is None:
            self.angle_offset = rng.random() * 2 * np.pi / self.n_actions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.is_eval = is_eval
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.discrete_action = False
        self.use_reward = False

    def step(self, action: np.ndarray) -> StepResult:
        if False or self.is_eval:
            reward = cosine_distance(action, self.target_vector)
        else:
            reward = -((action - self.target_vector) ** 2).sum() / 10
        return self.target_vector, reward, True, {"at_goal": True}

    def reset(self) -> np.ndarray:
        idx = rng.integers(0, self.n_actions)
        angle = self.angle_offset + 2 * np.pi * idx / self.n_actions
        self.target_vector = np.array([np.sin(angle), np.cos(angle)])
        return self.target_vector


class Virtual(gym.Env):
    def __init__(
        self,
        *,
        reward_structure: str,
        obs_type: str,
        single_step: bool,
        is_eval: bool,
        goal_radius: float,
        world_radius: float,
        max_step_scale: float,
        variant: str,
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__()
        self.goal_radius = goal_radius
        self.world_radius = world_radius
        self.single_step = single_step
        self.is_eval = is_eval
        self.max_step_scale = max_step_scale
        self.variant = variant

        self.max_steps = int(self.world_radius * self.max_step_scale)

        # TODO Add different distribution
        # TODO Add n-dimensional
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        # assert reward_structure in ("proximity", "none", "constant", "constant-only")
        assert reward_structure in ("cosine", "cosine-only", "constant")
        self.reward_structure = reward_structure

        # assert obs_type in ("vector", "direction")
        assert obs_type in ("direction",)
        self.obs_type = obs_type

        # For type purposes
        self.num_steps = 0
        self.stop = False
        self.location = np.zeros(self.observation_space.shape)

        # TODO Remove during cleanup, legacy variables
        self.discrete_action = False

    def _take_action(self, action: np.ndarray) -> None:
        try:
            assert type(action) == np.ndarray
            assert action.shape == (2,)
        except Exception as e:
            print(type(action))
            print(action)
            print(action.shape)
            raise e
        act_norm = get_norm(action)
        if act_norm > 1:
            action /= act_norm
        action /= self.world_radius
        self.location += action

    def _get_observation(self) -> np.ndarray:
        if self.obs_type == "direction":
            # Observation should never have norm 0 since it would be at the goal
            return -self.location / get_norm(self.location)
        else:
            raise NotImplementedError()

    def _get_step_result(self, action: np.ndarray) -> StepResult:
        info: Dict[str, Any] = {}

        observation = self._get_observation()

        at_goal = get_norm(self.location) <= self.goal_radius / self.world_radius
        if (self.num_steps > 0 and at_goal) or self.num_steps > self.max_steps:
            self.stop = True

        prev_vec = action - self.location
        # TODO Do we need to worry about cosine distance not checking magnitude?
        cosine_dist = cosine_distance(prev_vec, action)
        reward_scale = 1.0 if self.variant == "unscaled" else 0.01
        if self.single_step:
            reward = cosine_dist * reward_scale
            self.stop = self.num_steps > 0
            info["at_goal"] = reward
        elif self.is_eval:
            reward = float(at_goal)
            info["at_goal"] = at_goal
        else:
            info["at_goal"] = at_goal
            reward = 0.0
            if self.reward_structure in ("cosine", "cosine-only"):
                reward += (cosine_dist - 2) * reward_scale
            elif self.reward_structure == "constant":
                reward += -reward_scale
            if self.stop:
                if at_goal and self.reward_structure == "cosine":
                    reward += 1.0
                else:
                    reward += 0
        return observation, reward, self.stop, info

    def step(self, action: np.ndarray) -> StepResult:
        theta = np.pi / 4
        transform = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        # action = action @ transform
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        self.num_steps += 1
        # If the max has been reached, force the agent to stop
        prev_location = self.location.copy()
        self._take_action(action)
        obs, reward, done, info = self._get_step_result(action)
        obs = self._get_observation()
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        n_dim = 2
        # Pulled from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        u = rng.normal(0, 1, n_dim)
        norm = (u ** 2).sum() ** 0.5
        if self.variant == "triangle-init":
            # https://en.wikipedia.org/wiki/Triangular_distribution
            a = self.goal_radius / self.world_radius
            c = a
            b = 1.0
            r = rng.random()
            radius = b - np.sqrt((1 - r) * (b - a) * (b - c))
        else:
            radius = rng.uniform(self.goal_radius / self.world_radius, 1.0)
        self.location = radius * u / norm
        self.stop = False
        self.num_steps = 0
        dummy_action = np.zeros(self.observation_space.shape)
        return self._get_observation()
