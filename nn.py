from typing import List, Callable
from functools import partial

import numpy as np  # type: ignore
import gym  # type: ignore
from torch import nn
import torch


class BottleneckPolicy(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        *,
        bottleneck: str,
        pre_arch: List[int],
        post_arch: List[int],
        temp: float,
        bottleneck_hard: bool,
        act: str,
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__()

        if act == "tanh":
            activation: nn.Module = nn.Tanh()
        elif act == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f'Activation "{act}" not recognized.')

        # We need to make a copy of this bebcause stable baselines reuses references
        pre_arch = [x for x in pre_arch]
        pre_arch.insert(0, obs_space.shape[0])
        pre_layers: List[nn.Module] = []
        for i in range(len(pre_arch) - 1):
            if i != 0:
                pre_layers.append(activation)
            pre_layers.append(nn.Linear(pre_arch[i], pre_arch[i + 1]))
        self.pre_net = nn.Sequential(*pre_layers)

        self.temp = temp
        self.bottleneck_hard = bottleneck_hard
        if bottleneck == "none":
            self.bottleneck: Callable = torch.sigmoid
        elif bottleneck in ("sm", "softmax"):
            self.bottleneck = lambda x: nn.functional.softmax(x / self.temp, dim=-1)
        elif bottleneck in ("gsm", "gumbel-softmax"):
            self.bottleneck = partial(
                nn.functional.gumbel_softmax,
                tau=self.temp,
                hard=self.bottleneck_hard,
                dim=-1,
            )

        post_arch = [x for x in post_arch]
        post_arch.insert(0, pre_arch[-1])
        post_layers: List[nn.Module] = []
        for i in range(len(post_arch) - 1):
            if i != 0:
                post_layers.append(activation)
            post_layers.append(nn.Linear(post_arch[i], post_arch[i + 1]))
        self.post_net = nn.Sequential(*post_layers)

        self.features_dim = post_arch[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        x = self.bottleneck(x)
        x = self.post_net(x)
        return x

    def forward_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        x = self.bottleneck(x)
        return x
