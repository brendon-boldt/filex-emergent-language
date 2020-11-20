from typing import List, Any, Callable, Tuple, Optional, Dict, Union, Type

import gym  # type: ignore
from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import (  # type: ignore
    BaseFeaturesExtractor,
    is_image_space,
)


class Perceptron(nn.Module):
    def __init__(self, sizes: List[int]) -> None:
        super(self.__class__, self).__init__()
        self.sizes = sizes

        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            if i != len(sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> th.Tensor:
        x = self.model(x)
        return th.tanh(x)


class ScalableCnn(nn.Module):
    # def __init__(self, dim, out_size, input_lsize, ratio, channels) -> None:
    def __init__(self, obs_space: gym.spaces.Box, out_size: int = 0x10, ratio: int = 2):
        super(self.__class__, self).__init__()
        self.obs_space = obs_space
        # Maybe we'll get to 3 one day
        self.out_size = out_size
        # (floor . log_2) the clever way
        self.input_lsize = sum(
            self.obs_space.shape[-1] // 2 ** i >= 2 for i in range(20)
        )
        self.dim = 2
        self.features_dim = out_size * ratio

        if self.dim == 1:
            conv: Callable[..., nn.Module] = nn.Conv1d
            pool: Callable[..., nn.Module] = nn.AvgPool1d
        elif self.dim == 2:
            conv = nn.Conv2d
            pool = nn.AvgPool2d

        assert self.out_size >= self.input_lsize
        layers = []
        chans_in = self.obs_space.shape[0]
        for i in reversed(range(0, self.input_lsize)):
            chans_out = ratio * (self.out_size - i)
            layers.append(conv(chans_in, chans_out, 3, padding=1))
            layers.append(pool(2, 2))
            if i != 0:
                layers.append(nn.Tanh())
            chans_in = chans_out
        self.linears = nn.Sequential(
            nn.Tanh(),
            nn.Linear(chans_out, 0x40),
            nn.ReLU(),
            nn.Linear(0x40, features_dim),
        )
        self.cnn = nn.Sequential(*layers)

    def forward(self, x) -> th.Tensor:
        # x = x.view((1, 1, -1))
        # x.unsqueeze_(1)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.linears(x)
        # x = nn.functional.gumbel_softmax(x, tau=1)
        # x = nn.functional.softmax(x, dim=-1)
        # return x.transpose(2, 1)
        return x
