from typing import List, cast
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing_extensions import TypedDict
from gym import spaces
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from mbag.environment.blocks import MinecraftBlocks


class MbagModel(ABC, TorchModelV2):
    """
    A model to be used with MbagAutoregressiveActionDistribution.
    """

    @abstractmethod
    def action_type_model(self, dist_inputs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def block_id_model(
        self, dist_inputs: torch.Tensor, action_type: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def block_location_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
        block_id: torch.Tensor,
    ) -> torch.Tensor:
        ...


class MbagConvolutionalModelConfig(TypedDict, total=False):
    embedding_size: int
    num_layers: int
    filter_size: int
    hidden_channels: int


CONV_DEFAULT_CONFIG: MbagConvolutionalModelConfig = {
    "embedding_size": 8,
    "num_layers": 3,
    "filter_size": 3,
    "hidden_channels": 32,
}


class MbagConvolutionalModel(MbagModel, nn.Module):
    """
    Has an all-convolutional backbone and separate heads for each part of the
    autoregressive action distribution.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        obs_space = obs_space.original_space
        assert isinstance(obs_space, spaces.Tuple)
        self.world_obs_space: spaces.Box = obs_space[0]

        assert isinstance(action_space, spaces.Tuple)
        self.action_type_space: spaces.Discrete = action_space[0]
        self.block_location_space: spaces.Discrete = action_space[1]
        self.block_id_space: spaces.Discrete = action_space[2]

        assert self.block_location_space.n == np.prod(self.world_obs_space.shape[-3:])

        extra_config = CONV_DEFAULT_CONFIG
        extra_config.update(cast(MbagConvolutionalModelConfig, kwargs))
        self.embedding_size = extra_config["embedding_size"]
        self.num_layers = extra_config["num_layers"]
        self.filter_size = extra_config["filter_size"]
        self.hidden_channels = extra_config["hidden_channels"]
        self.in_planes = 2  # TODO: update if we add more

        self.block_id_embedding = nn.Embedding(
            num_embeddings=len(MinecraftBlocks.ID2NAME),
            embedding_dim=self.embedding_size,
        )

        self.vf_share_layers: bool = model_config["vf_share_layers"]
        if self.vf_share_layers:
            self.backbone = self._construct_backbone()
        else:
            self.action_backbone = self._construct_backbone()
            self.value_backbone = self._construct_backbone()

        self.action_type_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, self.action_type_space.n, bias=True),
        )
        self.block_id_head = nn.Sequential(
            nn.Linear(
                self.hidden_channels + self.action_type_space.n,
                self.embedding_size,
                bias=True,
            ),
        )
        self.block_location_head = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_channels
                + self.action_type_space.n
                + self.embedding_size,
                out_channels=1,
                kernel_size=1,
                stride=1,
            ),
        )

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def _construct_backbone(self):
        backbone_layers: List[nn.Module] = []
        for layer_index in range(self.num_layers):
            backbone_layers.append(
                nn.Conv3d(
                    in_channels=self.embedding_size * self.in_planes
                    if layer_index == 0
                    else self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.filter_size,
                    stride=1,
                    padding=(self.filter_size - 1) // 2,
                )
            )
            backbone_layers.append(nn.ReLU())
        return nn.Sequential(*backbone_layers)

    def forward(self, input_dict, state, seq_lens):
        (world_obs,) = input_dict["obs"]

        # TODO: embed other block info?
        world_obs = world_obs.long()
        embedded_blocks = self.block_id_embedding(world_obs[:, 0])
        embedded_goal_blocks = self.block_id_embedding(world_obs[:, 2])
        embedded_obs = torch.cat([embedded_blocks, embedded_goal_blocks], dim=-1)

        self._embedded_obs = embedded_obs.permute(0, 4, 1, 2, 3)

        if self.vf_share_layers:
            self._backbone_out = self.backbone(self._embedded_obs)
            return self._backbone_out, []
        else:
            return self.action_backbone(self._embedded_obs), []

    def action_type_model(self, dist_inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.action_type_head(dist_inputs))

    def block_id_model(
        self, dist_inputs: torch.Tensor, action_type: torch.Tensor
    ) -> torch.Tensor:
        avg_dist_inputs = F.adaptive_avg_pool3d(dist_inputs, (1, 1, 1)).flatten(
            start_dim=1
        )
        head_input = torch.cat(
            [avg_dist_inputs, F.one_hot(action_type, self.action_type_space.n)], dim=1
        )
        out_embedding = self.block_id_head(head_input)
        return cast(
            torch.Tensor, out_embedding @ self.block_id_embedding.weight.transpose(0, 1)
        )

    def block_location_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
        block_id: torch.Tensor,
    ) -> torch.Tensor:
        world_size = dist_inputs.size()[-3:]
        head_input = torch.cat(
            [
                dist_inputs,
                F.one_hot(action_type, self.action_type_space.n)[
                    :, :, None, None, None
                ].expand(-1, -1, *world_size),
                self.block_id_embedding(block_id)[:, :, None, None, None].expand(
                    -1, -1, *world_size
                ),
            ],
            dim=1,
        )
        return cast(torch.Tensor, self.block_location_head(head_input))

    def value_function(self):
        if self.vf_share_layers:
            return self.value_head(self._backbone_out)[:, 0]
        else:
            return self.value_head(self.value_backbone(self._embedded_obs))[:, 0]


ModelCatalog.register_custom_model("mbag_convolutional_model", MbagConvolutionalModel)
