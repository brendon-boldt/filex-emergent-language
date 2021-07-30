from functools import partial
from typing import Dict, List, Tuple, Type, Union, Any

from stable_baselines3.common.policies import ActorCriticPolicy  # type: ignore
from torch import nn  # type: ignore
import torch


class BottleneckExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Dict[str, Any],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super(self.__class__, self).__init__()

        # We need to make a copy of this bebcause stable baselines reuses references
        pre_arch = [x for x in net_arch["pre_bottleneck_arch"]]
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
        self.bottleneck = partial(
            nn.functional.gumbel_softmax,
            tau=self.temp,
            hard=self.bottleneck_hard,
            dim=-1,
        )

        post_arch = [x for x in net_arch["post_bottleneck_arch"]]
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

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_x = self.pre_net(features)
        pi_x = self.bottleneck(pi_x)
        pi_x = self.post_net(pi_x)
        vf_x = self.vf_net(features)
        return pi_x, vf_x

    def forward_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        x = self.bottleneck(x)
        return x


class BottleneckPolicy(ActorCriticPolicy):
    # This class may need to be modified if Stable Baselines is updated
    def _build_mlp_extractor(self) -> None:
        # This overrides the default MLP extractor.
        self.mlp_extractor = BottleneckExtractor(  # type: ignore
            self.features_dim,
            net_arch=self.net_arch,  # type: ignore
            activation_fn=self.activation_fn,
            device=self.device,
        )
