import torch
from torch import nn
import numpy as np

eps = 1e-8


class PerceptionNet(nn.Module):
    def __init__(self, log_scale, output_ratio):
        super(PerceptionNet, self).__init__()
        self.log_scale = log_scale
        self.output_ratio = output_ratio

        layers = []
        chans_in = 1
        for i in range(self.log_scale):
            if i == 0:
                chans_out = self.output_ratio * chans_in
            else:
                chans_out = 2 * chans_in
            layers.append(nn.Conv1d(chans_in, chans_out, 3, padding=1))
            layers.append(nn.AvgPool1d(2, 2))
            chans_in = chans_out
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(1, 1, -1)
        return self.model(x)


class ScalableEnv:
    def __init__(self, *, lsize: int, obs_lscale: int, action_scale: int) -> None:
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
        self.action_space = (-1.0, 1.0)

        # the largest number of loci the agent can move in one step
        self.action_scale = action_scale

    def get_observation(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.lscale >= self.obs_lscale:
            obs_loc = self.location // (self.size // self.obs_dim)
        else:
            raise NotImplementedError()
        obs[obs_loc] = 1.0
        return obs

    def get_perception(self) -> np.ndarray:
        obs = self.get_observation()

    def do_action(self, action: float) -> None:
        self.location += int(np.round(action * self.action_scale))
        self.location = np.clip(self.location, 0, self.size - 1)

    def check_goal(self) -> bool:
        pass

    def get_reward(self) -> float:
        pass


def sample_worlds():
    # 4x1 world
    small_discrete = ScalableEnv(lsize=2, obs_lscale=2, action_scale=1)
    medium_discrete = ScalableEnv(lsize=3, obs_lscale=3, action_scale=1)
    # 2 ** (7 - 5) == 4 steps to get across the world
    continuous_large_step = ScalableEnv(lsize=7, obs_lscale=7, action_scale=2 ** 5)
    # 2 ** 7 == 128 steps to get across the world
    continuous_small_step = ScalableEnv(lsize=7, obs_lscale=7, action_scale=1)

def test_world():
    env = ScalableEnv(lsize=2, obs_lscale=3, action_scale=2)
    print(env.get_observation())
    for i in range(10):
        env.do_action(0.8)
        print(env.get_observation())

def model_sizes():
    for i in range(1, 11):
        pn = PerceptionNet(i, 2)
        inp = torch.zeros(2 ** pn.log_scale)
        print(pn(inp).shape)
        print(np.sum([np.prod(x.size()) for x in pn.parameters()]))




if __name__ == "__main__":
    model_sizes()
