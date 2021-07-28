from typing import List, Tuple, cast
import gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn.functional as F
from torch import nn

from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)
from ray.rllib.models.catalog import ModelCatalog

from mbag.environment.types import MbagAction
from mbag.environment.blocks import MinecraftBlocks
from .torch_models import MbagModel


class MbagAutoregressiveActionDistribution(TorchDistributionWrapper):
    """
    An auto-regressive action distribution for the MBAG environment. It first samples
    an action type (e.g., NOOP, PLACE_BLOCK, etc.). Then, if necessary, it samples a
    block type given the action type. Finally, if necessary, it samples a
    block location given the action type and block type.

    This is meant to be used with the models in models.py
    """

    model: MbagModel
    inputs: torch.Tensor  # type: ignore
    model_device: torch.device
    inputs_device: torch.device

    _world_obs: torch.Tensor
    _cached_action_type = None
    _cached_action_type_logits = None
    _cached_block_location = None
    _cached_block_location_logits = None
    _cached_block_id_logits = None

    PLACEABLE_BLOCK_MASK = torch.tensor(
        [
            block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS
            for block_id in range(len(MinecraftBlocks.ID2NAME))
        ],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    def __init__(self, inputs: List[torch.Tensor], model: TorchModelV2):
        super().__init__(inputs, model)

        self._world_obs = getattr(self.model, "_world_obs", None)
        if self._world_obs is not None:
            self._world_obs = self._world_obs[: self.inputs.size()[0]]

        self.model_device = next(iter(cast(nn.Module, self.model).parameters())).device
        self.inputs_device = self.inputs.device
        self.inputs = self.inputs.to(self.model_device)

    def sample(self):
        # First, sample an action_type.
        action_type_dist = self._action_type_distribution()
        action_type = action_type_dist.sample()

        # Next, sample a block_location.
        block_location_dist = self._block_location_distribution(action_type)
        block_location = block_location_dist.sample()

        # Finally, sample a block_id.
        block_id_dist = self._block_id_distribution(action_type, block_location)
        block_id = block_id_dist.sample()

        self._sampled_logp = self._calculate_logp(
            action_type_dist,
            action_type,
            block_id_dist,
            block_id,
            block_location_dist,
            block_location,
        )
        return action_type, block_location, block_id

    def deterministic_sample(self):
        # First, sample an action_type.
        action_type_dist = self._action_type_distribution()
        action_type = action_type_dist.deterministic_sample()

        # Next, sample a block_location.
        block_location_dist = self._block_location_distribution(action_type)
        block_location = block_location_dist.deterministic_sample()

        # Finally, sample a block_id.
        block_id_dist = self._block_id_distribution(action_type, block_location)
        block_id = block_id_dist.deterministic_sample()

        self._sampled_logp = self._calculate_logp(
            action_type_dist,
            action_type,
            block_id_dist,
            block_id,
            block_location_dist,
            block_location,
        )
        return action_type, block_location, block_id

    def sampled_action_logp(self):
        return self._sampled_logp

    def logp(self, actions):
        actions = actions.to(self.model_device)

        action_type = actions[:, 0].long()
        block_location = actions[:, 1].long()
        block_id = actions[:, 2].long()

        action_type_dist, block_id_dist, block_location_dist = self._all_distributions(
            action_type, block_location
        )

        return self._calculate_logp(
            action_type_dist,
            action_type,
            block_id_dist,
            block_id,
            block_location_dist,
            block_location,
        )

    def entropy(self):
        if (
            self._cached_action_type is not None
            and self._cached_block_location is not None
        ):
            (
                action_type_dist,
                block_id_dist,
                block_location_dist,
            ) = self._all_distributions(
                self._cached_action_type, self._cached_block_location
            )
        else:
            action_type_dist = self._action_type_distribution()
            action_type = action_type_dist.sample()
            block_location_dist = self._block_location_distribution(action_type)
            block_location = block_location_dist.sample()
            block_id_dist = self._block_id_distribution(action_type, block_location)

        # Only count block_id and block_location entropy for actions which use them.
        block_id_use_prob = action_type_dist.dist.probs[
            :, MbagAction.BLOCK_ID_ACTION_TYPES
        ].sum(1)
        block_location_use_prob = action_type_dist.dist.probs[
            :, MbagAction.BLOCK_LOCATION_ACTION_TYPES
        ].sum(1)

        entropy = (
            action_type_dist.entropy()
            + block_id_dist.entropy() * block_id_use_prob
            + block_location_dist.entropy() * block_location_use_prob
        )
        if torch.any(entropy > 100):
            import pdb

            pdb.set_trace()
        return entropy

    def kl(self, other: "MbagAutoregressiveActionDistribution"):
        if (
            self._cached_action_type is not None
            and self._cached_block_location is not None
        ):
            (
                action_type_dist,
                block_id_dist,
                block_location_dist,
            ) = self._all_distributions(
                self._cached_action_type, self._cached_block_location
            )
        else:
            action_type_dist = self._action_type_distribution()
            action_type = action_type_dist.sample()
            block_location_dist = self._block_location_distribution(action_type)
            block_location = block_location_dist.sample()
            block_id_dist = self._block_id_distribution(action_type, block_location)

        action_type_kl = action_type_dist.kl(other._action_type_distribution())
        block_id_kl = block_id_dist.kl(
            other._block_id_distribution(action_type, block_location, skip_cache=True)
        )
        # Ignore infinite KL for block_id since the supervised loss tends to push
        # some probabilities down to zero.
        block_id_kl[torch.isinf(block_id_kl)] = 0
        block_location_kl = block_location_dist.kl(
            other._block_location_distribution(action_type, skip_cache=True)
        )

        return action_type_kl + block_id_kl + block_location_kl

    def _action_type_distribution(self) -> TorchCategorical:
        if self._cached_action_type_logits is None:
            self._cached_action_type_logits = self.model.action_type_model(self.inputs)
        return TorchCategorical(self._cached_action_type_logits)  # type: ignore

    def _block_location_distribution(
        self, action_type, mask_logit=-1e8, skip_cache=False
    ) -> TorchCategorical:
        if skip_cache:
            block_location_logits = self.model.block_location_model(
                self.inputs,
                action_type,
            )
        else:
            if self._cached_block_location_logits is None or not torch.all(
                self._cached_action_type == action_type
            ):
                self._cached_block_location_logits = self.model.block_location_model(
                    self.inputs,
                    action_type,
                )
                self._cached_action_type = action_type
            block_location_logits = self._cached_block_location_logits
        # Should be a BxWxHxD tensor:
        assert len(block_location_logits.size()) == 4
        return self._block_location_logits_to_distribution(
            action_type, block_location_logits.clone(), mask_logit
        )

    def _block_id_distribution(
        self, action_type, block_location, mask_logit=-1e8, skip_cache=False
    ) -> TorchCategorical:
        if skip_cache:
            block_id_logits = self.model.block_id_model(
                self.inputs, action_type, block_location
            )
        else:
            if self._cached_block_id_logits is None or not (
                torch.all(self._cached_action_type == action_type)
                and torch.all(self._cached_block_location == block_location)
            ):
                self._cached_block_id_logits = self.model.block_id_model(
                    self.inputs, action_type, block_location
                )
                self._cached_action_type = action_type
                self._cached_block_location = block_location
            block_id_logits = self._cached_block_id_logits

        # Mask out logits for placing unplaceable blocks.
        block_id_logits[
            (action_type == MbagAction.PLACE_BLOCK)[:, None]
            & ~MbagAutoregressiveActionDistribution.PLACEABLE_BLOCK_MASK[None, :]
        ] = mask_logit

        return TorchCategorical(block_id_logits)  # type: ignore

    def _block_location_logits_to_distribution(
        self, action_type, block_location_logits, mask_logit=-1e8
    ) -> TorchCategorical:
        if self._world_obs is not None:
            # Mask the distribution to blocks that can actually be affected.
            # First, we can't break air or bedrock.
            block_location_logits[
                (action_type == MbagAction.BREAK_BLOCK)[:, None, None, None]
                & (
                    (self._world_obs[:, 0] == MinecraftBlocks.AIR)
                    | (self._world_obs[:, 0] == MinecraftBlocks.BEDROCK)
                )
            ] = mask_logit

            # Next, we can only place in locations that are next to a solid block and
            # currently occupied by air.
            solid_block_ids = torch.tensor(
                list(MinecraftBlocks.SOLID_BLOCK_IDS),
                dtype=torch.uint8,
                device=self._world_obs.device,
            )
            solid_blocks = (
                self._world_obs[:, 0, :, :, :, None] == solid_block_ids
            ).any(-1)
            next_to_solid = (
                F.conv3d(
                    solid_blocks[:, None].float(),
                    torch.ones((1, 1, 3, 3, 3), device=solid_blocks.device),
                    padding=1,
                )
                > 0
            ).squeeze(1)
            block_location_logits[
                (action_type == MbagAction.PLACE_BLOCK)[:, None, None, None]
                & ((self._world_obs[:, 0] != MinecraftBlocks.AIR) | ~next_to_solid)
            ] = mask_logit

        return TorchCategorical(block_location_logits.flatten(start_dim=1))  # type: ignore

    def _all_distributions(
        self, action_type, block_location
    ) -> Tuple[TorchCategorical, TorchCategorical, TorchCategorical]:
        if (
            self._cached_action_type_logits is None
            or self._cached_block_location_logits is None
            or self._cached_block_id_logits is None
            or not (
                torch.all(self._cached_action_type == action_type)
                and torch.all(self._cached_block_location == block_location)
            )
        ):
            (
                self._cached_action_type_logits,
                self._cached_block_id_logits,
                self._cached_block_location_logits,
            ) = self.model.action_model(self.inputs, action_type, block_location)
            self._cached_action_type = action_type
            self._cached_block_location = block_location
        return (
            TorchCategorical(self._cached_action_type_logits),
            TorchCategorical(self._cached_block_id_logits),
            self._block_location_logits_to_distribution(
                action_type, self._cached_block_location_logits.clone()
            ),
        )

    def _calculate_logp(
        self,
        action_type_dist: TorchCategorical,
        action_type,
        block_id_dist: TorchCategorical,
        block_id,
        block_location_dist: TorchCategorical,
        block_location,
    ):
        """
        Carefully calculates the log probability of different actions, ignoring parts
        of the action which don't matter (for instance, block_location in a movement
        action).
        """

        action_type_logp = action_type_dist.logp(action_type)

        block_id_logp = block_id_dist.logp(block_id)
        block_id_action_types = torch.tensor(
            MbagAction.BLOCK_ID_ACTION_TYPES, device=block_id.device
        )
        block_id_mask = (action_type[:, None] == block_id_action_types).any(1)
        block_id_logp[~block_id_mask] = 0

        block_location_logp = block_location_dist.logp(block_location)
        block_location_action_types = torch.tensor(
            MbagAction.BLOCK_LOCATION_ACTION_TYPES, device=block_location.device
        )
        block_location_mask = (action_type[:, None] == block_location_action_types).any(
            1
        )
        block_location_logp[~block_location_mask] = 0

        # If for some reason some of the locations given to this method are invalid
        # (i.e., a block can't be placed/broken there), then just set the log probs
        # to 0.
        block_location_logp[
            block_location_dist.dist.probs.gather(-1, block_location[:, None])[:, 0]
            == 0
        ] = 0

        return (action_type_logp + block_id_logp + block_location_logp).to(
            self.inputs_device
        )

    @staticmethod
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ):
        return 1


ModelCatalog.register_custom_action_dist(
    "mbag_autoregressive", MbagAutoregressiveActionDistribution
)
