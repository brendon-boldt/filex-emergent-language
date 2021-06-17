from functools import partial
from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Callable, Any, cast

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import ActorCriticPolicy  # type: ignore
import numpy as np  # type: ignore
import gym  # type: ignore
from torch import nn
import torch
import torch as th
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import create_sde_features_extractor


class BottleneckExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        # net_arch: List[Union[int, Dict[str, List[int]]]],
        net_arch: Dict[str, Any],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        # *,
        # obs_space: gym.spaces.Box,
        # input_dim: gym.spaces.Box,
        # bottleneck: str,
        # pre_arch: List[int],
        # post_arch: List[int],
        # temp: float,
        # bottleneck_hard: bool,
        # act: str,
        # **kwargs,
    ) -> None:
        super(self.__class__, self).__init__()

        # if act == "tanh":
        #     activation: nn.Module = nn.Tanh()
        # elif act == "relu":
        #     activation = nn.ReLU()
        # else:
        #     raise ValueError(f'Activation "{act}" not recognized.')

        # We need to make a copy of this bebcause stable baselines reuses references
        pre_arch = [x for x in net_arch["pre_arch"]]
        # pre_arch.insert(0, obs_space.shape[0])
        pre_arch.insert(0, feature_dim)
        pre_layers: List[nn.Module] = []
        for i in range(len(pre_arch) - 1):
            if i != 0:
                pre_layers.append(activation_fn())
            pre_layers.append(nn.Linear(pre_arch[i], pre_arch[i + 1]))
        self.pre_net = nn.Sequential(*pre_layers)

        vf_layers: List[nn.Module] = []
        for i in range(len(pre_arch) - 1):
            if i != 0:
                vf_layers.append(activation_fn())
            vf_layers.append(nn.Linear(pre_arch[i], pre_arch[i + 1]))
        self.vf_net = nn.Sequential(*vf_layers)

        self.temp = net_arch["temp"]
        self.bottleneck_hard = net_arch["bottleneck_hard"]
        if net_arch["bottleneck"] == "none":
            self.bottleneck: Callable = torch.sigmoid
            # self.bottleneck: Callable = lambda x: x
            # self.bottleneck: Callable = torch.relu
        elif net_arch["bottleneck"] in ("sm", "softmax"):
            # self.bottleneck: Callable = torch.sigmoid
            self.bottleneck = lambda x: nn.functional.softmax(x / self.temp, dim=-1)
        elif net_arch["bottleneck"] in ("gsm", "gumbel-softmax"):
            self.bottleneck = partial(
                nn.functional.gumbel_softmax,
                tau=self.temp,
                hard=self.bottleneck_hard,
                dim=-1,
            )

        post_arch = [x for x in net_arch["post_arch"]]
        post_arch.insert(0, pre_arch[-1])
        post_layers: List[nn.Module] = []
        for i in range(len(post_arch) - 1):
            if i != 0:
                post_layers.append(activation_fn())
            post_layers.append(nn.Linear(post_arch[i], post_arch[i + 1]))
        post_layers.append(activation_fn())
        self.post_net = nn.Sequential(*post_layers)

        self.latent_dim_pi = post_arch[-1]
        self.latent_dim_vf = pre_arch[-1]

        # self.features_dim = post_arch[-1]

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_x = self.pre_net(features)
        pi_x = self.bottleneck(pi_x)
        pi_x = self.post_net(pi_x)
        vf_x = 0 * self.vf_net(features)
        return pi_x, vf_x

    def forward_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        x = self.bottleneck(x)
        return x


class MixedPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: (Callable) Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        # self.mlp_extractor = MlpExtractor(
        #     self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        # )
        bnx = BottleneckExtractor(
            self.features_dim,
            net_arch=cast(Any, self.net_arch),
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.mlp_extractor = cast(Any, bnx)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(  # type: ignore
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = (
                latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            )
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_sde_dim,
                log_std_init=self.log_std_init,
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))  # type: ignore

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore
