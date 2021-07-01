from typing import Tuple, Any, Dict, Union, cast, Optional

import gym  # type: ignore
from gym import spaces
import numpy as np  # type: ignore

StepResult = Tuple[Any, float, bool, Dict]

rng = np.random.default_rng()
GOLDEN_RATIO = (np.sqrt(5) + 1) / 2
GOLDEN_ANGLE = 2 * np.pi * (2 - GOLDEN_RATIO)


def get_norm(x):
    return np.sqrt((x ** 2).sum(-1))


def cosine_similarity(x, y) -> float:
    return (x * y).sum() / max(get_norm(x) * get_norm(y), 1e-5)


class NavToCenter(gym.Env):
    def __init__(
        self,
        *,
        base_reward_type: str,
        reward_shape_type: str,
        obs_type: str,
        is_eval: bool,
        goal_radius: float,
        world_radius: float,
        max_step_scale: float,
        half_life: float,
        rs_multiplier: float,
        reward_scale: float,
        variant: Optional[str],
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__()
        self.goal_radius = goal_radius
        self.world_radius = world_radius
        self.is_eval = is_eval
        self.max_step_scale = max_step_scale
        self.variant = variant
        self.half_life = half_life
        self.reward_scale = reward_scale
        self.rs_multiplier = rs_multiplier

        self.max_steps = int(self.world_radius * self.max_step_scale)

        assert obs_type in ("vector", "direction", "both")
        self.obs_type = obs_type

        # TODO Add different distribution
        # TODO Add n-dimensional
        if self.obs_type == "both":
            obs_shape = (4,)
        else:
            obs_shape = (2,)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        assert reward_shape_type in ("cosine", "l2")
        self.reward_shape_type = reward_shape_type

        assert base_reward_type in ("at-end", "every-step")
        self.base_reward_type = base_reward_type

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

    def get_observation(self) -> np.ndarray:
        direction = -self.location / get_norm(self.location)
        vector = -self.location
        if self.obs_type == "direction":
            # Observation should never have norm 0 since it would be at the goal
            return direction
        elif self.obs_type == "vector":
            return vector
        elif self.obs_type == "both":
            return np.concatenate([direction, vector])
        else:
            raise NotImplementedError()

    def _get_step_result(self, action: np.ndarray) -> StepResult:
        info: Dict[str, Any] = {}

        observation = self.get_observation()

        at_goal = get_norm(self.location) <= self.goal_radius / self.world_radius
        if (self.num_steps > 0 and at_goal) or self.num_steps > self.max_steps:
            self.stop = True

        if not self.is_eval and rng.random() > 2 ** (-1 / self.half_life):
            self.stop = True

        prev_vec = action - self.location
        cosine_sim = cosine_similarity(prev_vec, action)
        if self.is_eval:
            reward = float(at_goal)
            info["at_goal"] = at_goal
        else:
            info["at_goal"] = at_goal
            reward = 0.0
            if self.reward_shape_type == "cosine":
                reward += self.rs_multiplier * (cosine_sim - 1)
            elif self.reward_shape_type == "l2":
                raise NotImplementedError()
                # norm = get_norm(
                #     prev_vec / get_norm(prev_vec) - action * self.world_radius
                # )
                # reward = scale_factor * (reward_each_step + norm * self.rs_multiplier)
            if self.stop:
                if at_goal:
                    reward += 1.0
        return observation, reward, self.stop, info

    def step(self, action: np.ndarray) -> StepResult:
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        self.num_steps += 1
        # If the max has been reached, force the agent to stop
        prev_location = self.location.copy()
        self._take_action(action)
        obs, reward, done, info = self._get_step_result(action)
        obs = self.get_observation()
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        n_dim = 2
        # Pulled from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        u = rng.normal(0, 1, n_dim)
        norm = (u ** 2).sum() ** 0.5
        if not self.is_eval and self.variant == "triangle-init":
            # https://en.wikipedia.org/wiki/Triangular_distribution
            a = self.goal_radius / self.world_radius
            c = a
            b = 1.0
            r = rng.random()
            radius = b - np.sqrt((1 - r) * (b - a) * (b - c))
        else:
            radius = np.sqrt(
                rng.uniform((self.goal_radius / self.world_radius) ** 2, 1.0)
            )
        self.location = radius * u / norm
        self.stop = False
        self.num_steps = 0
        return self.get_observation()

    def fib_disc_init(self, i, n) -> np.ndarray:
        theta = i * GOLDEN_ANGLE
        r = np.sqrt(i / n)
        if r > self.goal_radius:
            raise ValueError(f"Index i={i} is too small (within the goal radius).")
        self.location = r * np.array([np.cos(theta), np.sin(theta)])
        return self.get_observation()
