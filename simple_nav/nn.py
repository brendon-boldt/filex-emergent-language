from functools import partial
from typing import Dict, List, Tuple, Type, Union, Any

from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy  # type: ignore
from torch import nn  # type: ignore
import torch
import numpy as np


class BottleneckExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Dict[str, Any],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super(self.__class__, self).__init__()

        self.signal_game = net_arch["signal_game"]
        self.n_opts = net_arch["n_opts"]
        # TODO Very hacky and fragile
        if self.signal_game:
            self.n_dims = (feature_dim - self.n_opts) // self.n_opts
        else:
            self.n_dims = feature_dim

        # We need to make a copy of this bebcause stable baselines reuses references
        pre_arch = [x for x in net_arch["pre_bottleneck_arch"]]
        pre_arch.insert(0, self.n_dims)
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
        if self.signal_game:
            post_input_size = pre_arch[-1] + self.n_dims * self.n_opts
        else:
            post_input_size = pre_arch[-1]
        post_arch.insert(0, post_input_size)
        post_layers: List[nn.Module] = []
        for i in range(len(post_arch) - 1):
            if i != 0:
                post_layers.append(activation_fn())
            post_layers.append(nn.Linear(post_arch[i], post_arch[i + 1]))
        post_layers.append(activation_fn())
        self.post_net = nn.Sequential(*post_layers)

        self.latent_dim_pi = post_arch[-1]
        self.latent_dim_vf = pre_arch[-1]

        self.backward_log: List = []

        self.bn_activations: List[np.ndarray] = []

    def _restructure(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        correct_idxs = x[:, :self.n_opts].argmax(-1)
        vecs = x[:, self.n_opts:].reshape(x.shape[0], self.n_opts, -1)
        return correct_idxs, vecs

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.signal_game:
            correct_idxs, vecs = self._restructure(features)
            ar = torch.arange(vecs.shape[0], dtype=torch.long)
            features = vecs[ar, correct_idxs]
        pi_x = self.pre_net(features)
        self._logits = pi_x.detach().cpu()
        pi_x = self.bottleneck(pi_x)
        self._bn_activations = pi_x.detach().cpu()
        if self.signal_game:
            pi_x = torch.cat([pi_x, torch.flatten(vecs, 1)], dim=-1)
        pi_x = self.post_net(pi_x)
        vf_x = self.vf_net(features)
        return pi_x, vf_x


# This method may need to be modified if Stable Baselines is updated
def _build_mlp_extractor(self) -> None:
    self.mlp_extractor = BottleneckExtractor(  # type: ignore
        self.features_dim,
        net_arch=self.net_arch,  # type: ignore
        activation_fn=self.activation_fn,
        device=self.device,
    )

class BottleneckPolicy(ActorCriticPolicy):
    _build_mlp_extractor = _build_mlp_extractor


class MultiInputBottleneckPolicy(MultiInputActorCriticPolicy):
    _build_mlp_extractor = _build_mlp_extractor
