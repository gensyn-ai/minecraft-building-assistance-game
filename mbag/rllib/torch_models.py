from typing import List, Optional, Tuple, cast
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
    def block_location_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def block_id_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
        block_location: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def action_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
        block_location: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This should return the distribution outputs for all three parts of the action.
        For some model implementations, this may be computationally cheaper than
        computing each one in sequence when doing optimization.
        """

        action_type_logits = self.action_type_model(dist_inputs)
        block_location_logits = self.block_location_model(dist_inputs, action_type)
        block_id_logits = self.block_id_model(dist_inputs, action_type, block_location)
        return action_type_logits, block_id_logits, block_location_logits


class MbagConvolutionalModelConfig(TypedDict, total=False):
    embedding_size: int
    use_extra_features: bool
    """Use extra hand-designed features as input to the network."""
    num_conv_1_layers: int
    """Number of 1x1x1 convolutions before the main backbone."""
    num_layers: int
    filter_size: int
    hidden_channels: int
    num_block_id_layers: int
    """Number of extra layers for the block ID head."""
    num_location_layers: int
    """Number of extra layers for the block location head."""


CONV_DEFAULT_CONFIG: MbagConvolutionalModelConfig = {
    "embedding_size": 8,
    "use_extra_features": False,
    "num_conv_1_layers": 0,
    "num_layers": 3,
    "filter_size": 3,
    "hidden_channels": 32,
    "num_block_id_layers": 1,
    "num_location_layers": 1,
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
        self.use_extra_features = extra_config["use_extra_features"]
        self.num_conv_1_layers = extra_config["num_conv_1_layers"]
        self.num_layers = extra_config["num_layers"]
        self.filter_size = extra_config["filter_size"]
        self.hidden_channels = extra_config["hidden_channels"]
        self.num_block_id_layers = extra_config["num_block_id_layers"]
        self.num_location_layers = extra_config["num_location_layers"]

        self.in_planes = 2  # TODO: update if we add more
        self.in_channels = self.in_planes * self.embedding_size
        if self.use_extra_features:
            self.in_channels += 1

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

        block_location_in_channels = self.hidden_channels + self.action_type_space.n
        block_location_layers: List[nn.Module] = []
        for layer_index in range(self.num_location_layers):
            block_location_layers.append(
                nn.Conv3d(
                    in_channels=block_location_in_channels
                    if layer_index == 0
                    else self.hidden_channels,
                    out_channels=1
                    if layer_index == self.num_location_layers - 1
                    else self.hidden_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
            if layer_index < self.num_location_layers - 1:
                block_location_layers.append(nn.ReLU())
        self.block_location_head = nn.Sequential(*block_location_layers)

        block_id_in_channels = self.hidden_channels + self.action_type_space.n
        block_id_layers: List[nn.Module] = []
        for layer_index in range(self.num_block_id_layers):
            block_id_layers.append(
                nn.Linear(
                    block_id_in_channels if layer_index == 0 else self.hidden_channels,
                    self.block_id_space.n
                    if layer_index == self.num_block_id_layers - 1
                    else self.hidden_channels,
                )
            )
            if layer_index < self.num_block_id_layers - 1:
                block_id_layers.append(nn.ReLU())
        self.block_id_head = nn.Sequential(*block_id_layers)

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def _construct_backbone(self):
        backbone_layers: List[nn.Module] = []
        for layer_index in range(self.num_conv_1_layers + self.num_layers):
            if layer_index < self.num_conv_1_layers:
                filter_size = 1
            else:
                filter_size = self.filter_size
            backbone_layers.append(
                nn.Conv3d(
                    in_channels=self.in_channels
                    if layer_index == 0
                    else self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=filter_size,
                    stride=1,
                    padding=(filter_size - 1) // 2,
                )
            )
            backbone_layers.append(nn.ReLU())
        return nn.Sequential(*backbone_layers)

    def forward(self, input_dict, state, seq_lens):
        (self._world_obs,) = input_dict["obs"]

        # TODO: embed other block info?
        self._world_obs = self._world_obs.long()
        embedded_blocks = self.block_id_embedding(self._world_obs[:, 0])
        embedded_goal_blocks = self.block_id_embedding(self._world_obs[:, 2])
        embedded_obs_pieces = [embedded_blocks, embedded_goal_blocks]
        if self.use_extra_features:
            # Feature for if goal block is the same as the current block at each
            # location.
            embedded_obs_pieces.append(
                (self._world_obs[:, 0] == self._world_obs[:, 2]).float()[..., None]
            )
        embedded_obs = torch.cat(embedded_obs_pieces, dim=-1)

        self._embedded_obs = embedded_obs.permute(0, 4, 1, 2, 3)

        if self.vf_share_layers:
            self._backbone_out = self.backbone(self._embedded_obs)
            self._backbone_out_shape = self._backbone_out.size()[1:]
            return self._backbone_out.flatten(start_dim=1), []
        else:
            backbone_out = self.action_backbone(self._embedded_obs)
            self._backbone_out_shape = backbone_out.size()[1:]
            return backbone_out.flatten(start_dim=1), []

    def action_type_model(self, dist_inputs: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.action_type_head(
                dist_inputs.reshape((-1,) + self._backbone_out_shape)
            ),
        )

    def block_location_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
    ) -> torch.Tensor:
        dist_inputs = dist_inputs.reshape((-1,) + self._backbone_out_shape)
        world_size = dist_inputs.size()[-3:]
        head_input = torch.cat(
            [
                dist_inputs,
                F.one_hot(action_type, self.action_type_space.n)[
                    :, :, None, None, None
                ].expand(-1, -1, *world_size),
            ],
            dim=1,
        )
        return cast(torch.Tensor, self.block_location_head(head_input)[:, 0])

    def block_id_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
        block_location: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = dist_inputs.size()[0]
        head_input = torch.cat(
            [
                dist_inputs.reshape((-1,) + self._backbone_out_shape).flatten(
                    start_dim=2
                )[torch.arange(batch_size), :, block_location],
                F.one_hot(action_type, self.action_type_space.n),
            ],
            dim=1,
        )
        return cast(torch.Tensor, self.block_id_head(head_input))

    def value_function(self):
        if self.vf_share_layers:
            return self.value_head(self._backbone_out).squeeze(1)
        else:
            return self.value_head(self.value_backbone(self._embedded_obs)).squeeze(1)


ModelCatalog.register_custom_model("mbag_convolutional_model", MbagConvolutionalModel)


class MbagTransformerModelConfig(TypedDict, total=False):
    embedding_size: int
    position_embedding_size: int
    num_layers: int
    num_heads: int
    hidden_size: int
    num_decoder_layers: int


TRANSFORMER_DEFAULT_CONFIG: MbagTransformerModelConfig = {
    "embedding_size": 10,
    "position_embedding_size": 12,
    "num_layers": 3,
    "num_heads": 2,
    "hidden_size": 32,
    "num_decoder_layers": 2,
}


class MbagTransformerModel(MbagModel, nn.Module):
    """
    Concatenates all observation parts into one long sequence, with different encodings
    for each part, and then applies a transformer to the result. There are special
    positions in the sequence for the action type output and the block ID output.
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
        self.world_size = self.world_obs_space.shape[-3:]

        assert isinstance(action_space, spaces.Tuple)
        self.action_type_space: spaces.Discrete = action_space[0]
        self.block_location_space: spaces.Discrete = action_space[1]
        self.block_id_space: spaces.Discrete = action_space[2]

        assert self.block_location_space.n == np.prod(self.world_size)

        extra_config = TRANSFORMER_DEFAULT_CONFIG
        extra_config.update(cast(MbagTransformerModelConfig, kwargs))
        self.embedding_size = extra_config["embedding_size"]
        self.position_embedding_size = extra_config["position_embedding_size"]
        self.num_layers = extra_config["num_layers"]
        self.num_heads = extra_config["num_heads"]
        self.hidden_size = extra_config["hidden_size"]
        self.num_decoder_layers = extra_config["num_decoder_layers"]
        assert (
            self.embedding_size * 2 + self.position_embedding_size <= self.hidden_size
        )

        self.block_id_embedding = nn.Embedding(
            num_embeddings=len(MinecraftBlocks.ID2NAME),
            embedding_dim=self.embedding_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(self.world_size + (self.position_embedding_size,))
        )

        # Special embeddings for the sequence elements for value function, action_type,
        # block_id, and block_location outputs.
        self.vf_embedding = nn.Parameter(torch.zeros((1, self.position_embedding_size)))
        self.action_part_embeddings = nn.Parameter(
            torch.zeros((3, self.position_embedding_size))
        )

        # Initialize positional embeddings along each dimension.
        dim_embedding_size = self.position_embedding_size // 3
        self.position_embedding.data[
            ..., :dim_embedding_size
        ] = self._get_position_embedding(
            self.position_embedding.size()[0],
            dim_embedding_size,
        )[
            :, None, None
        ]
        self.position_embedding.data[
            ..., dim_embedding_size : dim_embedding_size * 2
        ] = self._get_position_embedding(
            self.position_embedding.size()[0],
            dim_embedding_size,
        )[
            None, :, None
        ]
        self.position_embedding.data[
            ..., dim_embedding_size * 2 : dim_embedding_size * 3
        ] = self._get_position_embedding(
            self.position_embedding.size()[0],
            dim_embedding_size,
        )[
            None, None, :
        ]

        self.vf_embedding.data.normal_()
        self.action_part_embeddings.data.normal_()
        self.action_part_embeddings.data[:, -1] = 1

        self.vf_share_layers: bool = model_config["vf_share_layers"]
        if not self.vf_share_layers:
            raise ValueError(
                "MbagTransformerModel does not support vf_share_layers=False"
            )

        self.backbone_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size,
                batch_first=True,
            ),
            self.num_layers,
        )
        self.action_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size,
                batch_first=True,
            ),
            self.num_decoder_layers,
        )

        self.action_type_head = nn.Linear(
            self.hidden_size, self.action_type_space.n, bias=True
        )
        self.block_location_head = nn.Linear(self.hidden_size, 1, bias=False)
        self.block_id_head = nn.Linear(
            self.hidden_size, self.block_id_space.n, bias=True
        )
        self.value_head = nn.Linear(self.hidden_size, 1, bias=True)

    def _get_position_embedding(self, seq_len: int, size: int) -> torch.Tensor:
        embedding = torch.zeros(seq_len, size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, size, 2).float() * (-np.log(10000.0) / size)
        )
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding

    def _pad_to_hidden_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads the last dimension of the given tensor so that it is self.hidden_size.
        """

        return F.pad(x, [0, self.hidden_size - x.size()[-1]])

    def forward(self, input_dict, state, seq_lens):
        (self._world_obs,) = input_dict["obs"]
        batch_size = self._world_obs.size()[0]

        # TODO: embed other block info?
        self._world_obs = self._world_obs.long()
        embedded_blocks = self.block_id_embedding(self._world_obs[:, 0])
        embedded_goal_blocks = self.block_id_embedding(self._world_obs[:, 2])

        embedded_world_obs = torch.cat(
            [
                self.position_embedding[None].expand(batch_size, -1, -1, -1, -1),
                embedded_blocks,
                embedded_goal_blocks,
            ],
            dim=-1,
        )
        flattened_world_obs = (
            embedded_world_obs.permute(0, 4, 1, 2, 3)
            .flatten(start_dim=2)
            .transpose(1, 2)
        )
        encoder_input = torch.cat(
            [
                self._pad_to_hidden_size(
                    self.vf_embedding[None].expand(batch_size, -1, -1)
                ),
                self._pad_to_hidden_size(flattened_world_obs),
            ],
            dim=1,
        )

        backbone_out = self.backbone_encoder(encoder_input)
        self._value_out = backbone_out[:, 0]
        self._backbone_out = backbone_out[:, 1:]

        return self._backbone_out, []

    def action_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: Optional[torch.Tensor] = None,
        block_location: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = dist_inputs.size()[0]
        decoder_inputs: List[torch.Tensor] = []

        # First input is just the action_type embedding.
        decoder_inputs.append(
            self._pad_to_hidden_size(
                self.action_part_embeddings[None, None, 0].expand(batch_size, -1, -1)
            )
        )

        if action_type is not None:
            # Second input is the chosen action_type.
            decoder_inputs.append(
                self._pad_to_hidden_size(
                    torch.cat(
                        [
                            self.action_part_embeddings[None, None, 1].expand(
                                batch_size, -1, -1
                            ),
                            F.one_hot(action_type, self.action_type_space.n)[:, None],
                        ],
                        dim=2,
                    )
                )
            )

        decoder_inputs.append(dist_inputs)
        decoder_seq_len = sum(inp.size()[1] for inp in decoder_inputs)
        target_mask = torch.zeros(
            (decoder_seq_len, decoder_seq_len),
            dtype=torch.bool,
            device=dist_inputs.device,
        )
        target_mask[0, 1:] = True  # action_type only depends on itself
        action_decoder_out = self.action_decoder(
            torch.cat(decoder_inputs, dim=1),
            dist_inputs,
            target_mask,
        )

        # Now, we pass the output through the appropriate heads.
        action_type_logits = self.action_type_head(action_decoder_out[:, 0])

        if action_type is not None:
            block_location_out = action_decoder_out[:, 2:]
            block_location_logits = self.block_location_head(block_location_out)[
                :, :, 0
            ]
        else:
            block_location_logits = torch.zeros(
                (batch_size, self.block_location_space.n), device=dist_inputs.device
            )

        if block_location is not None:
            block_id_head_input = action_decoder_out[
                torch.arange(batch_size),
                block_location + 2,
            ]
            block_id_logits = self.block_id_head(block_id_head_input)
        else:
            block_id_logits = torch.zeros(
                (batch_size, self.block_id_space.n), device=dist_inputs.device
            )

        block_location_logits = block_location_logits.reshape(-1, *self.world_size)

        return action_type_logits, block_id_logits, block_location_logits

    def action_type_model(self, dist_inputs: torch.Tensor) -> torch.Tensor:
        return self.action_model(dist_inputs)[0]

    def block_location_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_model(dist_inputs, action_type)[2]

    def block_id_model(
        self,
        dist_inputs: torch.Tensor,
        action_type: torch.Tensor,
        block_location: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_model(dist_inputs, action_type, block_location)[1]

    def value_function(self):
        return self.value_head(self._value_out)[:, 0]


ModelCatalog.register_custom_model("mbag_transformer_model", MbagTransformerModel)
