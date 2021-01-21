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


# Spirals outward
DISCRETE_ACT_DICT = {
    0: [0, 0],
    1: [1, 0],
    2: [0, 1],
    3: [-1, 0],
    4: [0, -1],
    5: [1, 1],
    6: [-1, 1],
    7: [-1, -1],
    8: [1, -1],
    9: [2, 0],
    10: [0, 2],
    11: [-2, 0],
    12: [0, -2],
}


class Scalable(gym.Env):
    def __init__(
        self,
        pixel_space: bool,
        discrete_action: bool,
        reward_structure: str,
        action_noise: float,
        obs_type: str,
        env_shape: str,
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
        self.discrete_action = discrete_action
        if discrete_action:
            if action_scale == 1:
                n_act = 5
            elif action_scale == 1.5:
                n_act = 9
            elif action_scale == 2:
                n_act = 13
            else:
                raise ValueError(
                    f"Cannot handle discrete action scale '{action_scale}'"
                )
            self.action_space = spaces.Discrete(n_act)
        else:
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

        assert reward_structure in ("proximity", "none", "constant", "constant-only")
        self.reward_structure = reward_structure

        assert obs_type in ("vector", "direction")
        self.obs_type = obs_type

        self.action_noise = action_noise

        assert env_shape in ("circle", "square")
        self.env_shape = env_shape

        # For type purposes
        self.num_steps = 0
        self.stop = False

    def _take_action(self, action: Union[np.int64, np.ndarray]) -> None:
        if self.discrete_action:
            assert type(action) == np.int64
            final_action = np.array(DISCRETE_ACT_DICT[action])
        else:
            assert type(action) == np.ndarray
            assert action.shape == (2,)  # type: ignore
            if not self.is_eval:
                action += rng.normal(0, self.action_noise, 2)
            action = np.round(action * self.action_scale)
            act_norm = get_norm(action)
            if act_norm > self.action_scale:
                action *= self.action_scale / act_norm
            final_action = action.astype(np.int32)
        self.location = np.clip(
            self.location + final_action,
            0,
            self.size - 1,
        )

    def _is_at_goal(self) -> bool:
        diff = (self.location - self.goal_location) / self.size
        dist = (diff ** 2).sum() ** 0.5
        return dist < (self.action_scale / self.size)

    def _get_observation(self, prev_location: np.ndarray) -> StepResult:
        info: Dict[str, Any] = {}
        if self.pixel_space:
            _observation = np.zeros(self.observation_space.shape)
            T = tuple
            _observation[T(self.location)][0] = 1.0
            # Distinguish passable locations from 0 padding
            # _observation[..., 1] = 0.1
            _observation[T(self.goal_location)][1] = 1.0
            observation = (_observation * 255).astype(np.uint8)
        else:
            observation = (self.goal_location - self.location) / self.size
            if self.obs_type == "direction":
                observation /= max(get_norm(observation), 1e-5)

        diff = (self.location - self.goal_location) / self.size
        dist = (diff ** 2).sum() ** 0.5
        prev_diff = (prev_location - self.goal_location) / self.size
        prev_dist = (prev_diff ** 2).sum() ** 0.5

        at_goal = self._is_at_goal()
        if (self.num_steps > 0 and at_goal) or self.num_steps > self.max_steps:
            self.stop = True

        prev_vec = self.goal_location - prev_location
        cur_vec = self.location - prev_location
        cosine_dist = cosine_distance(prev_vec, cur_vec)
        if self.single_step:
            reward = cosine_dist
            # reward = 1.0 if dist < prev_dist else -1.0
            self.stop = self.num_steps > 0
            info["at_goal"] = reward
        elif self.is_eval:
            reward = float(self.stop and at_goal)
            info["at_goal"] = at_goal
        else:
            info["at_goal"] = at_goal
            reward = 0.0
            if self.reward_structure == "proximity":
                # reward += -0.01 if dist < prev_dist else -0.02
                reward += -0.01 * (2 - cosine_dist)
            elif self.reward_structure in ("constant", "constant-only"):
                reward += -0.01
            elif self.reward_structure == "none":
                pass
            if self.stop:
                if at_goal and self.reward_structure != "constant-only":
                    reward += 1.0
                else:
                    reward += 0
        return observation, reward, self.stop, info

    def step(self, action: np.ndarray) -> StepResult:
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        self.num_steps += 1
        # If the max has been reached, force the agent to stop
        prev_location = self.location.copy()
        self._take_action(action)
        return self._get_observation(prev_location)

    def _in_circle(self, x) -> bool:
        return ((x - (self.size - 1) / 2) ** 2).sum() < (self.size / 2) ** 2

    def _is_valid_start(self) -> bool:
        if self.env_shape == "circle":
            return (
                not self._is_at_goal()
                and self._in_circle(self.goal_location)
                and self._in_circle(self.location)
            )
        else:
            return not self._is_at_goal()

    def reset(self) -> np.ndarray:
        self.location = rng.integers(0, self.size, (2,))
        self.goal_location = rng.integers(0, self.size, (2,))
        while not self._is_valid_start():
            self.location = rng.integers(0, self.size, (2,))
            self.goal_location = rng.integers(0, self.size, (2,))
        self.stop = False
        self.num_steps = 0
        return self._get_observation(self.location)[0]


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
        if get_norm(action) > 1:
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
        if self.single_step:
            reward = cosine_dist
            self.stop = self.num_steps > 0
            info["at_goal"] = reward
        elif self.is_eval:
            reward = float(at_goal)
            info["at_goal"] = at_goal
        else:
            info["at_goal"] = at_goal
            reward = 0.0
            if self.reward_structure in ("cosine", "cosine-only"):
                reward += -0.01 * (2 - cosine_dist)
            elif self.reward_structure == "constant":
                reward += -0.01
            if self.stop:
                if at_goal and self.reward_structure == "cosine":
                    reward += 1.0
                else:
                    reward += 0
        return observation, reward, self.stop, info

    def _apply_randomization(self) -> None:
        # TODO Remove
        if self.variant == "random-angle" and not self.is_eval:
            norm = get_norm(self.location)
            new_angle = rng.uniform(np.pi * 2)
            self.location = norm * np.array([np.sin(new_angle), np.cos(new_angle)])

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
        self._apply_randomization()
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
