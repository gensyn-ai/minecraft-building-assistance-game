import gym
from ray.rllib.utils.typing import ModelConfigDict
import torch
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)
from ray.rllib.models.catalog import ModelCatalog

from mbag.environment.types import MbagAction
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

    def sample(self):
        # First, sample an action_type.
        action_type_dist = self._action_type_distribution()
        action_type = action_type_dist.sample()

        # Next, sample a block_id.
        block_id_dist = self._block_id_distribution(action_type)
        block_id = block_id_dist.sample()

        # Finally, sample a block_location.
        block_location_dist = self._block_location_distribution(action_type, block_id)
        block_location = block_location_dist.sample()

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

        # Next, sample a block_id.
        block_id_dist = self._block_id_distribution(action_type)
        block_id = block_id_dist.deterministic_sample()

        # Finally, sample a block_location.
        block_location_dist = self._block_location_distribution(action_type, block_id)
        block_location = block_location_dist.deterministic_sample()

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
        action_type = actions[:, 0].long()
        block_location = actions[:, 1].long()
        block_id = actions[:, 2].long()

        action_type_dist = self._action_type_distribution()
        block_id_dist = self._block_id_distribution(action_type)
        block_location_dist = self._block_location_distribution(action_type, block_id)

        return self._calculate_logp(
            action_type_dist,
            action_type,
            block_id_dist,
            block_id,
            block_location_dist,
            block_location,
        )

    def entropy(self):
        action_type_dist = self._action_type_distribution()
        action_type = action_type_dist.sample()
        block_id_dist = self._block_id_distribution(action_type)
        block_id = block_id_dist.sample()
        block_location_dist = self._block_location_distribution(action_type, block_id)

        return (
            action_type_dist.entropy()
            + block_id_dist.entropy()
            + block_location_dist.entropy()
        )

    def kl(self, other: "MbagAutoregressiveActionDistribution"):
        action_type_dist = self._action_type_distribution()
        action_type = action_type_dist.sample()
        block_id_dist = self._block_id_distribution(action_type)
        block_id = block_id_dist.sample()
        block_location_dist = self._block_location_distribution(action_type, block_id)

        return (
            action_type_dist.kl(other._action_type_distribution())
            + block_id_dist.kl(other._block_id_distribution(action_type))
            + block_location_dist.kl(
                other._block_location_distribution(action_type, block_id)
            )
        )

    def _action_type_distribution(self) -> TorchCategorical:
        action_type_logits = self.model.action_type_model(self.inputs)
        return TorchCategorical(action_type_logits)  # type: ignore

    def _block_id_distribution(self, action_type) -> TorchCategorical:
        block_id_logits = self.model.block_id_model(self.inputs, action_type)
        return TorchCategorical(block_id_logits)  # type: ignore

    def _block_location_distribution(self, action_type, block_id) -> TorchCategorical:
        # Should be a BxWxHxD tensor:
        block_location_logits = self.model.block_location_model(
            self.inputs, action_type, block_id
        )
        return TorchCategorical(block_location_logits.flatten(start_dim=1))  # type: ignore

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
        block_id_mask = (block_id[:, None] == block_id_action_types).any(1)
        block_id_logp[~block_id_mask] = 0

        block_location_logp = block_location_dist.logp(block_location)
        block_location_action_types = torch.tensor(
            MbagAction.BLOCK_LOCATION_ACTION_TYPES, device=block_location.device
        )
        block_location_mask = (
            block_location[:, None] == block_location_action_types
        ).any(1)
        block_location_logp[~block_location_mask] = 0

        return action_type_logp + block_id_logp + block_location_logp

    @staticmethod
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ):
        return 1


ModelCatalog.register_custom_action_dist(
    "mbag_autoregressive", MbagAutoregressiveActionDistribution
)
