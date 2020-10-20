from typing import List

import gym  # type: ignore
from torch import nn  # type: ignore
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
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> th.Tensor:
        x = self.model(x)
        return x

class Cnn1D(nn.Module):
    def __init__(self, out_size, input_lsize, ratio, channels) -> None:
        super(self.__class__, self).__init__()
        self.out_size = out_size
        self.input_lsize = input_lsize

        assert self.out_size >= self.input_lsize
        layers = []
        chans_in = channels
        for i in reversed(range(0, self.input_lsize)):
            chans_out = ratio * (self.out_size - i)
            layers.append(nn.Conv1d(chans_in, chans_out, 3, padding=1))
            layers.append(nn.AvgPool1d(2, 2))
            chans_in = chans_out
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> th.Tensor:
        # x = x.view((1, 1, -1))
        # x.unsqueeze_(1)
        x = self.model(x)
        return x.transpose(2, 1)

class BasicCnn(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(BasicCnn, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space), (
            "You should use NatureCNN "
            f"only with images not with {observation_space} "
            "(you are probably using `CnnPolicy` instead of `MlpPolicy`)"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            # nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.Conv2d(n_input_channels, 8, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            # nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class SimpleCnn(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(SimpleCnn, self).__init__(observation_space, features_dim)
        assert is_image_space(observation_space)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 0x20, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(0x20, 0x20, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
