import numpy as np
from scipy import ndimage

from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import MbagAction, MbagObs, CURRENT_BLOCKS
from mbag.environment.blocks import MinecraftBlocks


class MbagActionDistribution(object):
    """
    Currently, this class contains utilities for construction distributions over
    environment actions.
    """

    PLACEABLE_BLOCK_MASK = np.array(
        [
            block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS
            for block_id in range(len(MinecraftBlocks.ID2NAME))
        ],
    )
    SOLID_BLOCK_IDS = np.array(
        list(MinecraftBlocks.SOLID_BLOCK_IDS),
        dtype=np.uint8,
    )

    @staticmethod
    def get_action_type_location_mask(
        config: MbagConfigDict, obs: MbagObs
    ) -> np.ndarray:
        """
        Given an environment configuration and a batch of observations, return a
        boolean NumPy array of shape
            (batch_size, NUM_ACTION_TYPES, width, height, depth)
        where valid combinations of action type and block location are True and
        invalid combinations are False.
        """

        world_obs, inventory_obs = obs
        batch_size, _, width, height, depth = world_obs.shape

        mask = np.ones(
            (batch_size, MbagAction.NUM_ACTION_TYPES, width, height, depth),
            dtype=np.bool8,
        )

        # Mask the distribution to blocks that can actually be affected.
        # First, we can't break air or bedrock.
        mask[:, MbagAction.BREAK_BLOCK][
            (
                (world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.AIR)
                | (world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.BEDROCK)
            )
        ] = False

        # Next, we can only place in locations that are next to a solid block and
        # currently occupied by air.
        solid_blocks = (
            world_obs[:, CURRENT_BLOCKS, :, :, :, None]
            == MbagActionDistribution.SOLID_BLOCK_IDS
        ).any(-1)
        next_to_solid = (
            ndimage.convolve(
                solid_blocks,
                np.ones((1, 3, 3, 3)),
                mode="constant",
            )
            > 0
        )
        mask[:, MbagAction.PLACE_BLOCK][
            (world_obs[:, CURRENT_BLOCKS] != MinecraftBlocks.AIR) | ~next_to_solid
        ] = False

        return mask

    @staticmethod
    def get_action_type_location_unique(
        config: MbagConfigDict, obs: MbagObs
    ) -> np.ndarray:
        """
        Given an environment configuration and a batch of observations, return an
        integer NumPy array of shape
            (batch_size, NUM_ACTION_TYPES, width, height, depth)
        where each nonzero element corresponds to a unique action; actions which have
        the same nonzero value in the array are equivalent and those which
        have different nonzero values are different.
        """

        world_obs, inventory_obs = obs
        batch_size = world_obs.shape[0]

        mask = MbagActionDistribution.get_action_type_location_mask(config, obs)
        unique = np.zeros(mask.shape, dtype=np.int32)

        # TODO: which actions are actually unique?
        unique_flat = unique.reshape((batch_size, -1))
        unique_flat[:] = np.arange(unique_flat.shape[1], dtype=unique.dtype)[None] + 1

        unique[~mask] = 0

        return unique

    @staticmethod
    def get_block_id_mask(
        config: MbagConfigDict,
        obs: MbagObs,
        action_type: np.ndarray,
        block_location: np.ndarray,
    ) -> np.ndarray:
        """
        Given an environment configuration, a batch of observations, and a corresponding
        batch of action types and block locations return an ndarray of shape
            (batch_size, NUM_BLOCKS)
        where valid block IDs to use in an action with the chosen actions types and
        block locations are True and invalid block IDs are False.
        """

        world_obs, inventory_obs = obs
        batch_size = world_obs.shape[0]

        mask = np.ones((batch_size, MinecraftBlocks.NUM_BLOCKS), dtype=np.bool8)

        # Can't place unplaceable block types.
        mask[
            (action_type == MbagAction.PLACE_BLOCK)[:, None]
            & ~MbagActionDistribution.PLACEABLE_BLOCK_MASK[None, :]
        ] = False

        return mask

    @staticmethod
    def get_block_id_unique(
        config: MbagConfigDict,
        obs: MbagObs,
        action_type: np.ndarray,
        block_location: np.ndarray,
    ) -> np.ndarray:
        """
        Given an environment configuration, a batch of observations, and a corresponding
        batch of action types and block locations return an integer ndarray of shape
            (batch_size, NUM_BLOCKS)
        with the same semantics as get_action_type_location_unique.
        """

        mask = MbagActionDistribution.get_block_id_mask(
            config, obs, action_type, block_location
        )
        unique = np.zeros(mask.shape, dtype=np.int32)

        # TODO: which actions are actually unique?
        unique[:] = np.arange(unique.shape[1], dtype=unique.dtype)[None] + 1

        unique[~mask] = 0

        return unique
