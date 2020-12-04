from typing import List, Any, Callable, Tuple, Optional, Dict, Union, Type
from functools import partial

import numpy as np  # type: ignore
import gym  # type: ignore
from torch import nn
import torch
from stable_baselines3.common.torch_layers import (  # type: ignore
    BaseFeaturesExtractor,
    is_image_space,
)
from stable_baselines3.common.policies import ActorCriticPolicy


class Perceptron(nn.Module):
    # def __init__(self, sizes: List[int]) -> None:
    def __init__(
        self, feature_dim: int, last_layer_dim_pi: int, last_layer_dim_vf: int
    ) -> None:
        super(self.__class__, self).__init__()
        self.sizes = [0x40]
        # self.sizes = sizes
        sizes = [feature_dim] + self.sizes

        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        self.last_layer_pi = nn.Linear(sizes[-1], last_layer_dim_pi)
        self.last_layer_vf = nn.Linear(sizes[-1], last_layer_dim_vf)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.model(x)
        return self.last_layer_pi(x), self.last_layer_vf(x)
        # return torch.tanh(x)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        # I don't know if ignoring the type here is okay
        self.mlp_extractor = Perceptron(self.features_dim)  # type: ignore


class BottleneckPolicy(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        bottleneck: str,
        pre_arch: List[int],
        post_arch: List[int],
        temp: float = 1.0,
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__()

        # We need to make a copy of this bebcause stable baselines reuses references
        pre_arch = [x for x in pre_arch]
        pre_arch.insert(0, obs_space.shape[0])
        pre_layers: List[nn.Module] = []
        for i in range(len(pre_arch) - 1):
            if i != 0:
                pre_layers.append(nn.Tanh())
            pre_layers.append(nn.Linear(pre_arch[i], pre_arch[i + 1]))
        self.pre_net = nn.Sequential(*pre_layers)

        self.temp = temp
        if bottleneck == "none":
            # self.bottleneck: Callable = nn.Identity()
            self.bottleneck: Callable = nn.functional.sigmoid
        elif bottleneck in ("sm", "softmax"):
            self.bottleneck = lambda x: nn.functional.softmax(x / self.temp, dim=-1)
        elif bottleneck in ("gsm", "gumbel-softmax"):
            self.bottleneck = partial(
                nn.functional.gumbel_softmax, tau=self.temp, dim=-1
            )

        post_arch = [x for x in post_arch]
        post_arch.insert(0, pre_arch[-1])
        post_layers: List[nn.Module] = []
        for i in range(len(post_arch) - 1):
            if i != 0:
                post_layers.append(nn.Tanh())
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


class ScalableCnn(nn.Module):
    # def __init__(self, dim, out_size, input_lsize, ratio, channels) -> None:
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        bottleneck: str,
        pre_arch: List[int],
        post_arch: List[int],
        out_size: int = 0x10,
        ratio: int = 2,
        **kwargs,
    ):
        super(self.__class__, self).__init__()
        self.obs_space = obs_space
        # Maybe we'll get to 3 one day
        self.out_size = out_size
        # (floor . log_2) the clever way
        self.input_lsize = sum(
            self.obs_space.shape[-1] // 2 ** i >= 2 for i in range(20)
        )

        conv = nn.Conv2d
        pool = nn.AvgPool2d
        # pool = nn.MaxPool2d

        assert self.out_size >= self.input_lsize
        layers: List[nn.Module] = []
        chans_in = self.obs_space.shape[0]
        for i in reversed(range(0, self.input_lsize)):
            chans_out = ratio * (self.out_size - i)
            layers.append(conv(chans_in, chans_out, 3, padding=1))
            layers.append(pool(2, 2))
            if i != 0:
                layers.append(nn.Tanh())
            chans_in = chans_out

        self.cnn = nn.Sequential(*layers)

        # Haha
        self.bottle_net = BottleneckPolicy(
            # We only need it for the shape
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(chans_out,)),
            bottleneck,
            pre_arch,
            post_arch,
        )
        self.features_dim = self.bottle_net.features_dim

    def forward(self, x) -> torch.Tensor:
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.bottle_net(x)
        return x

    def forward_bottleneck(self, x) -> torch.Tensor:
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.forward_bottleneck(x)
        return x
