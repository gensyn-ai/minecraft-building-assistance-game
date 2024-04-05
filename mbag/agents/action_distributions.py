import logging
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Union, cast

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    import torch

from mbag.environment.actions import MbagAction, MbagActionTuple, MbagActionType
from mbag.environment.blocks import MAX_PLAYER_REACH, MinecraftBlocks
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import (
    CURRENT_BLOCKS,
    CURRENT_PLAYER,
    NO_ONE,
    PLAYER_LOCATIONS,
    MbagObs,
)

logger = logging.getLogger(__name__)


class MbagActionDistribution(object):
    """
    Currently, this class contains utilities for construction distributions over
    environment actions.

    In general, a distribution over environment actions can be represented as an
    array of shape
        (NUM_CHANNELS, width, height, depth)
    whose elements sum to 1 such that the element at (action_type_block_id, x, y, z)
    corresponds to the probability of taking an action with the given action_type
    and block_id (more on this in the next paragraph) at the location x, y, z.

    The first dimension, action_type_block_id, combines the action_type and block_id
    fields of an action. Actions which do not use a block_id have a single channel
    for them and actions which do use a block_id have NUM_BLOCKS channels for them.
    That is, the channels are NOOP, PLACE_BLOCK air, PLACE_BLOCK bedrock,
    PLACE_BLOCK dirt, ..., BREAK_BLOCK, MOVE_POS_X, ..., GIVE_BLOCK air,
    GIVE_BLOCK bedrock, GIVE_BLOCK dirt, ..., GIVE_BLOCK wool. The number of channels
    is NUM_CHANNELS = NUM_ACTION_TYPES + len(BLOCK_ID_ACTION_TYPES) * (NUM_BLOCKS - 1).

    Alternatively, a distribution over environment actions can be represented as a flat
    distribution over a number of actions N. The value of N depends on the environment
    configuration since the valid actions are different depending on the environment.
    N counts width * height * depth for actions_type_block_ids which require a block
    location, but otherwise counts only 1 for those which don't.
    """

    # See below class definition for where CHANNELS is initialized.
    CHANNELS: List[Tuple[MbagActionType, int]] = []
    NUM_CHANNELS: int = MbagAction.NUM_ACTION_TYPES + len(
        MbagAction.BLOCK_ID_ACTION_TYPES
    ) * (MinecraftBlocks.NUM_BLOCKS - 1)

    NOOP = 0
    PLACE_BLOCK = slice(1, 1 + MinecraftBlocks.NUM_BLOCKS)
    BREAK_BLOCK = 1 + MinecraftBlocks.NUM_BLOCKS
    MOVE_POS_X = 2 + MinecraftBlocks.NUM_BLOCKS
    MOVE_NEG_X = 3 + MinecraftBlocks.NUM_BLOCKS
    MOVE_POS_Y = 4 + MinecraftBlocks.NUM_BLOCKS
    MOVE_NEG_Y = 5 + MinecraftBlocks.NUM_BLOCKS
    MOVE_POS_Z = 6 + MinecraftBlocks.NUM_BLOCKS
    MOVE_NEG_Z = 7 + MinecraftBlocks.NUM_BLOCKS
    GIVE_BLOCK = slice(
        8 + MinecraftBlocks.NUM_BLOCKS, 8 + 2 * MinecraftBlocks.NUM_BLOCKS
    )
    ACTION_TYPE2CHANNEL: Dict[MbagActionType, Union[int, slice]] = {
        MbagAction.NOOP: NOOP,
        MbagAction.PLACE_BLOCK: PLACE_BLOCK,
        MbagAction.BREAK_BLOCK: BREAK_BLOCK,
        MbagAction.MOVE_POS_X: MOVE_POS_X,
        MbagAction.MOVE_NEG_X: MOVE_NEG_X,
        MbagAction.MOVE_POS_Y: MOVE_POS_Y,
        MbagAction.MOVE_NEG_Y: MOVE_NEG_Y,
        MbagAction.MOVE_POS_Z: MOVE_POS_Z,
        MbagAction.MOVE_NEG_Z: MOVE_NEG_Z,
        MbagAction.GIVE_BLOCK: GIVE_BLOCK,
    }

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
    def get_valid_action_types(config: MbagConfigDict) -> Set[MbagActionType]:
        action_types = {MbagAction.NOOP, MbagAction.PLACE_BLOCK, MbagAction.BREAK_BLOCK}
        if not config["abilities"]["teleportation"]:
            action_types |= set(MbagAction.MOVE_ACTION_TYPES)
        if not config["abilities"]["inf_blocks"]:
            action_types.add(MbagAction.GIVE_BLOCK)
        return action_types

    @staticmethod
    def get_action_mapping(config: MbagConfigDict) -> np.ndarray:
        """
        Given env config, returns a numpy array of shape (N, 3) which maps flat
        actions to actions with (action_type, block_location_index, block_id)
        components.
        """

        mapping_parts: List[np.ndarray] = []
        valid_action_types = MbagActionDistribution.get_valid_action_types(config)
        for action_type, channel in MbagActionDistribution.ACTION_TYPE2CHANNEL.items():
            if action_type in valid_action_types:
                num_block_ids = (
                    MinecraftBlocks.NUM_BLOCKS
                    if action_type in MbagAction.BLOCK_ID_ACTION_TYPES
                    else 1
                )
                width, height, depth = config["world_size"]
                num_block_location_indices = (
                    width * height * depth
                    if action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES
                    else 1
                )
                block_ids_grid, block_location_grid = np.meshgrid(
                    np.arange(num_block_ids),
                    np.arange(num_block_location_indices),
                    indexing="ij",
                )

                mapping_part = np.empty(
                    (num_block_ids * num_block_location_indices, 3), dtype=int
                )
                mapping_part[:, 0] = action_type
                mapping_part[:, 1] = block_location_grid.flat
                mapping_part[:, 2] = block_ids_grid.flat
                mapping_parts.append(mapping_part)

        return cast(np.ndarray, np.concatenate(mapping_parts, axis=0))

    @staticmethod
    def get_flat_action(
        config: MbagConfigDict,
        action: MbagActionTuple,
    ) -> int:
        """
        Get the flattened ID of an action.
        """

        action_type, block_location, block_id = action
        if action_type not in MbagAction.BLOCK_ID_ACTION_TYPES:
            block_id = 0
        if action_type not in MbagAction.BLOCK_LOCATION_ACTION_TYPES:
            block_location = 0

        flat_id = 0

        valid_action_types = MbagActionDistribution.get_valid_action_types(config)
        for other_action_type in valid_action_types:
            num_block_ids = (
                MinecraftBlocks.NUM_BLOCKS
                if other_action_type in MbagAction.BLOCK_ID_ACTION_TYPES
                else 1
            )
            width, height, depth = config["world_size"]
            num_block_location_indices = (
                width * height * depth
                if other_action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES
                else 1
            )
            if other_action_type < action_type:
                flat_id += num_block_ids * num_block_location_indices
            elif other_action_type == action_type:
                flat_id += block_id * num_block_location_indices + block_location

        return flat_id

    @staticmethod
    def to_flat(
        config: MbagConfigDict, probs: np.ndarray, reduction=np.sum
    ) -> np.ndarray:
        """
        Given env config and a distribution array of shape
            (batch_size, NUM_CHANNELS, width, height, depth)
        returns an array of size (batch_size, N) which contains the flattened
        action distribution probabilities.
        """

        batch_size = probs.shape[0]
        valid_action_types = MbagActionDistribution.get_valid_action_types(config)
        flat_pieces: List[np.ndarray] = []
        for action_type, channel in MbagActionDistribution.ACTION_TYPE2CHANNEL.items():
            if action_type in valid_action_types:
                if action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES:
                    flat_piece = probs[:, channel].reshape((batch_size, -1))
                else:
                    flat_piece = reduction(
                        probs[:, channel], axis=(-3, -2, -1)
                    ).reshape((batch_size, -1))
                flat_pieces.append(flat_piece)
        return cast(np.ndarray, np.concatenate(flat_pieces, axis=1))

    @staticmethod
    def to_flat_torch(
        config: MbagConfigDict, probs: "torch.Tensor", reduction=None
    ) -> "torch.Tensor":
        import torch

        if reduction is None:
            reduction = torch.sum
        assert reduction is torch.sum or reduction is torch.logsumexp

        batch_size = probs.size()[0]

        valid_action_types = MbagActionDistribution.get_valid_action_types(config)
        flat_pieces: List[torch.Tensor] = []

        channels_to_reduce = [
            channel
            for action_type, channel in MbagActionDistribution.ACTION_TYPE2CHANNEL.items()
            if action_type not in MbagAction.BLOCK_LOCATION_ACTION_TYPES
        ]
        reduced_pieces = reduction(probs[:, channels_to_reduce], dim=(-3, -2, -1))

        for action_type, channel in MbagActionDistribution.ACTION_TYPE2CHANNEL.items():
            if action_type in valid_action_types:
                if action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES:
                    flat_piece = probs[:, channel].flatten(start_dim=1)
                else:
                    flat_piece = reduced_pieces[
                        :, channels_to_reduce.index(channel)
                    ].reshape((batch_size, -1))
                flat_pieces.append(flat_piece)
        return torch.cat(flat_pieces, dim=1)

    @staticmethod
    def to_flat_torch_logits(
        config: MbagConfigDict, logits: "torch.Tensor"
    ) -> "torch.Tensor":
        import torch

        return MbagActionDistribution.to_flat_torch(
            config, logits, reduction=torch.logsumexp
        )

    @staticmethod
    def get_mask(
        config: MbagConfigDict,
        obs: MbagObs,
        *,
        line_of_sight_masking=False,
        force_python_impl=False,
    ) -> np.ndarray:
        """
        Given an environment configuration and a batch of observations, return a
        boolean NumPy array of shape
            (batch_size, NUM_CHANNELS, width, height, depth)
        where valid actions are True and invalid actions are False.

        If line_of_sight_masking is False, then whether the player has line of sight
        to place/break blocks is not considered. If line_of_sight_masking is True, then
        it is considered, but this requires the C implementation to be installed and
        may be slower.
        """

        world_obs, inventory_obs, timestep = obs
        batch_size, _, width, height, depth = world_obs.shape

        if not force_python_impl:
            try:
                import _mbag

                mask = np.zeros(
                    (
                        batch_size,
                        MbagActionDistribution.NUM_CHANNELS,
                        width,
                        height,
                        depth,
                    ),
                    dtype=np.bool8,
                )
                world_obs = world_obs.astype(np.uint8)
                inventory_obs = inventory_obs.astype(np.int32)

                try:
                    for batch_index in range(batch_size):
                        mask[batch_index] = _mbag.get_action_distribution_mask(
                            world_obs[batch_index],
                            inventory_obs[batch_index],
                            int(timestep[batch_index]),
                            teleportation=config["abilities"]["teleportation"],
                            inf_blocks=config["abilities"]["inf_blocks"],
                            line_of_sight_masking=line_of_sight_masking,
                        )
                    return mask
                except RuntimeError as error:
                    if error.args[0] == "No player location found":
                        logger.warn("no player locations found in observation")
                        return mask
                    raise
            except ImportError:
                logger.warn(
                    "C implementation of get_mask not found, falling back to Python"
                )

        if line_of_sight_masking:
            raise ValueError(
                "line of sight masking is not supported in the Python implementation"
            )

        mask = np.ones(
            (batch_size, MbagActionDistribution.NUM_CHANNELS, width, height, depth),
            dtype=np.bool8,
        )

        # Mask invalid actions.
        valid_action_types = MbagActionDistribution.get_valid_action_types(config)
        for action_type, channel in MbagActionDistribution.ACTION_TYPE2CHANNEL.items():
            if action_type not in valid_action_types:
                mask[:, channel] = False

        # We can't break air or bedrock.
        mask[:, MbagActionDistribution.BREAK_BLOCK][
            (
                (world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.AIR)
                | (world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.BEDROCK)
            )
        ] = False

        # Can't place air or bedrock.
        for block_id in range(MinecraftBlocks.NUM_BLOCKS):
            if block_id not in MinecraftBlocks.PLACEABLE_BLOCK_IDS:
                mask[:, MbagActionDistribution.PLACE_BLOCK][:, block_id] = False

        # Next, we can only place in locations that are next to a solid block and
        # currently occupied by air.
        solid_blocks = (
            world_obs[:, CURRENT_BLOCKS, :, :, :, None]
            == MbagActionDistribution.SOLID_BLOCK_IDS
        ).any(-1)
        next_to_solid_filter = np.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ]
        )
        next_to_solid = (
            ndimage.convolve(
                solid_blocks,
                next_to_solid_filter[None],
                mode="constant",
            )
            > 0
        )
        invalid_place = (
            (world_obs[:, CURRENT_BLOCKS] != MinecraftBlocks.AIR)
            | ~next_to_solid
            | (world_obs[:, PLAYER_LOCATIONS] != NO_ONE)
        )
        mask[:, MbagActionDistribution.PLACE_BLOCK][
            np.repeat(invalid_place[:, None], MinecraftBlocks.NUM_BLOCKS, axis=1)
        ] = False

        # Next, we can only give blocks to locations with players in them
        mask[:, MbagActionDistribution.GIVE_BLOCK] &= (
            world_obs[:, None, PLAYER_LOCATIONS] != NO_ONE
        ) & (world_obs[:, None, PLAYER_LOCATIONS] != CURRENT_PLAYER)
        mask[:, MbagActionDistribution.GIVE_BLOCK, :, 1:, :] &= (
            world_obs[:, None, PLAYER_LOCATIONS, :, :-1, :]
            != world_obs[:, None, PLAYER_LOCATIONS, :, 1:, :]
        )

        if (
            np.all(world_obs[:, PLAYER_LOCATIONS] == 0)
            and not config["abilities"]["teleportation"]
        ):
            # Happens during loss intialization.
            logger.warn("no player locations found in observation")
        elif not config["abilities"]["teleportation"]:
            # If we can't teleport, then we can only place or break blocks up to 3 blocks away
            player_location = world_obs[:, PLAYER_LOCATIONS] == CURRENT_PLAYER
            batch_indices, player_x, player_y, player_z = np.nonzero(player_location)
            _, feet_filter = np.unique(batch_indices, return_index=True)
            feet_x = np.full(batch_size, np.iinfo(np.int32).min, dtype=np.int32)
            feet_y = np.full(batch_size, np.iinfo(np.int32).min, dtype=np.int32)
            feet_z = np.full(batch_size, np.iinfo(np.int32).min, dtype=np.int32)
            batch_indices = batch_indices[feet_filter]
            feet_x[batch_indices] = player_x[feet_filter]
            feet_y[batch_indices] = player_y[feet_filter]
            feet_z[batch_indices] = player_z[feet_filter]
            head_x, head_y, head_z = feet_x, feet_y + 1, feet_z

            world_x, world_y, world_z = np.meshgrid(
                np.arange(width), np.arange(height), np.arange(depth), indexing="ij"
            )
            dist_from_player = np.sqrt(
                (world_x[None] - head_x[:, None, None, None]) ** 2
                + (world_y[None] - head_y[:, None, None, None]) ** 2
                + (world_z[None] - head_z[:, None, None, None]) ** 2
            )
            reachable = dist_from_player <= MAX_PLAYER_REACH

            mask[:, MbagActionDistribution.BREAK_BLOCK] &= reachable
            mask[:, MbagActionDistribution.PLACE_BLOCK] &= reachable[:, None]

            # If we can't teleport, then we can only give blocks to from one block away from players
            conv_mask = np.ones((1, 3, 4, 3))
            conv_mask[0, 1, 1:2, 1] = 0
            reachable_1 = (
                (np.abs(world_x[None] - head_x[:, None, None, None]) <= 1)
                & (np.abs(world_y[None] - head_y[:, None, None, None]) <= 1)
                & (np.abs(world_z[None] - head_z[:, None, None, None]) <= 1)
            )
            mask[:, MbagActionDistribution.GIVE_BLOCK] &= reachable_1[:, None]

            # We can only move in directions that are not blocked by solid blocks
            # or players.
            mask[:, MbagActionDistribution.MOVE_POS_X] = (
                MbagActionDistribution._is_valid_position_to_move_to(
                    config, world_obs, feet_x + 1, feet_y, feet_z
                )[:, None, None, None]
            )
            mask[:, MbagActionDistribution.MOVE_NEG_X] = (
                MbagActionDistribution._is_valid_position_to_move_to(
                    config, world_obs, feet_x - 1, feet_y, feet_z
                )[:, None, None, None]
            )
            mask[:, MbagActionDistribution.MOVE_POS_Y] = (
                MbagActionDistribution._is_valid_position_to_move_to(
                    config, world_obs, feet_x, feet_y + 1, feet_z
                )[:, None, None, None]
            )
            mask[:, MbagActionDistribution.MOVE_NEG_Y] = (
                MbagActionDistribution._is_valid_position_to_move_to(
                    config, world_obs, feet_x, feet_y - 1, feet_z
                )[:, None, None, None]
            )
            mask[:, MbagActionDistribution.MOVE_POS_Z] = (
                MbagActionDistribution._is_valid_position_to_move_to(
                    config, world_obs, feet_x, feet_y, feet_z + 1
                )[:, None, None, None]
            )
            mask[:, MbagActionDistribution.MOVE_NEG_Z] = (
                MbagActionDistribution._is_valid_position_to_move_to(
                    config, world_obs, feet_x, feet_y, feet_z - 1
                )[:, None, None, None]
            )

        if not config["abilities"]["inf_blocks"]:
            # If we don't have infinite blocks, we can only place or give blocks we
            # have.
            have_blocks = inventory_obs[:, 0] > 0
            have_blocks = np.tile(
                have_blocks[:, :, None, None, None], (1, 1) + world_obs.shape[-3:]
            )
            for channels in [
                MbagActionDistribution.PLACE_BLOCK,
                MbagActionDistribution.GIVE_BLOCK,
            ]:
                mask[:, channels][~have_blocks] = False

        return mask

    @staticmethod
    def _is_valid_position_to_move_to(
        config: MbagConfigDict,
        world_obs: np.ndarray,
        new_feet_x: np.ndarray,
        new_feet_y: np.ndarray,
        new_feet_z: np.ndarray,
    ) -> np.ndarray:
        """
        Returns True if the given position is a valid position to move to.
        """

        width, height, depth = config["world_size"]
        batch_size = world_obs.shape[0]

        # Specialize for when batch_size is 1.
        if batch_size == 1:
            new_feet_x = new_feet_x[0]
            new_feet_y = new_feet_y[0]
            new_feet_z = new_feet_z[0]
            if (
                new_feet_x < 0
                or new_feet_x >= width
                or new_feet_y < 0
                or new_feet_y >= height
                or new_feet_z < 0
                or new_feet_z >= depth
            ):
                valid_bool = False
            else:
                feet_world_obs = world_obs[0, :, new_feet_x, new_feet_y, new_feet_z]
                head_world_obs = world_obs[
                    0, :, new_feet_x, min(new_feet_y + 1, height - 1), new_feet_z
                ]
                valid_bool = (
                    (feet_world_obs[CURRENT_BLOCKS] == MinecraftBlocks.AIR)
                    and (head_world_obs[CURRENT_BLOCKS] == MinecraftBlocks.AIR)
                    and (
                        (feet_world_obs[PLAYER_LOCATIONS] == NO_ONE)
                        or (feet_world_obs[PLAYER_LOCATIONS] == CURRENT_PLAYER)
                    )
                    and (
                        (head_world_obs[PLAYER_LOCATIONS] == NO_ONE)
                        or (head_world_obs[PLAYER_LOCATIONS] == CURRENT_PLAYER)
                    )
                )
            return np.array([valid_bool])

        valid = (
            (new_feet_x >= 0)
            & (new_feet_x < width)
            & (new_feet_y >= 0)
            & (new_feet_y < height)
            & (new_feet_z >= 0)
            & (new_feet_z < depth)
        )

        feet_world_obs = world_obs[
            np.arange(batch_size)[valid],
            :,
            new_feet_x[valid],
            new_feet_y[valid],
            new_feet_z[valid],
        ]
        head_world_obs = world_obs[
            np.arange(batch_size)[valid],
            :,
            new_feet_x[valid],
            np.minimum(new_feet_y[valid] + 1, height - 1),
            new_feet_z[valid],
        ]

        valid[valid] &= (
            (feet_world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.AIR)
            & (head_world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.AIR)
            & (
                (feet_world_obs[:, PLAYER_LOCATIONS] == NO_ONE)
                | (feet_world_obs[:, PLAYER_LOCATIONS] == CURRENT_PLAYER)
            )
            & (
                (head_world_obs[:, PLAYER_LOCATIONS] == NO_ONE)
                | (head_world_obs[:, PLAYER_LOCATIONS] == CURRENT_PLAYER)
            )
        )

        return valid

    @staticmethod
    def get_mask_flat(
        config: MbagConfigDict,
        obs: MbagObs,
        *,
        line_of_sight_masking=False,
    ) -> np.ndarray:
        mask = MbagActionDistribution.get_mask(
            config, obs, line_of_sight_masking=line_of_sight_masking
        )
        return MbagActionDistribution.to_flat(config, mask, reduction=np.all)


for action_type in MbagAction.ACTION_TYPES:
    if action_type in MbagAction.BLOCK_ID_ACTION_TYPES:
        for block_id, _ in enumerate(MinecraftBlocks.ID2NAME):
            MbagActionDistribution.CHANNELS.append((action_type, block_id))
    else:
        MbagActionDistribution.CHANNELS.append((action_type, 0))
assert len(MbagActionDistribution.CHANNELS) == MbagActionDistribution.NUM_CHANNELS
