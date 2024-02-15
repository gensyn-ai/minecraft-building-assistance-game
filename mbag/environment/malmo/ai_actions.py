from typing import List, NamedTuple, Optional, Union

from ..actions import MbagAction
from ..blocks import MinecraftBlocks
from ..config import MbagConfigDict
from ..types import WorldLocation, get_block_counts_in_inventory
from .malmo_state import (
    BlockDiff,
    InventoryDiff,
    LocationDiff,
    MalmoState,
    MalmoStateDiff,
)


class MalmoPlaceBreakAIAction(NamedTuple):
    action: MbagAction
    yaw: float
    pitch: float
    player_location: Optional[WorldLocation]
    """
    When teleportation=True, this should be set to the location where the player
    should teleport to before placing or breaking the block. When teleportation=False,
    this should be None.
    """


class MalmoMoveAIAction(NamedTuple):
    action: MbagAction
    new_location: WorldLocation


class MalmoGiveAIAction(NamedTuple):
    action: MbagAction
    recipient_player_index: int
    blocks_to_give: int


MalmoAIAction = Union[MalmoPlaceBreakAIAction, MalmoMoveAIAction, MalmoGiveAIAction]


def get_state_diffs_for_ai_action(
    malmo_state: MalmoState,
    player_index: int,
    ai_action: MalmoAIAction,
    env_config: MbagConfigDict,
) -> List[MalmoStateDiff]:
    """
    Gets the state diffs that are expected by running this AI action. This allows
    us to ignore those state diffs and not try to interpret them as human actions.
    """

    block_counts = get_block_counts_in_inventory(
        malmo_state.player_inventories[player_index]
    )

    if isinstance(ai_action, MalmoPlaceBreakAIAction):
        if ai_action.action.action_type == MbagAction.PLACE_BLOCK:
            block_id = ai_action.action.block_id
            prev_count = block_counts[block_id]
            state_diffs: List[MalmoStateDiff] = [
                BlockDiff(
                    ai_action.action.block_location,
                    MinecraftBlocks.AIR,
                    block_id,
                ),
            ]
            if not env_config["abilities"]["inf_blocks"]:
                state_diffs.append(
                    InventoryDiff(
                        player_index,
                        block_id,
                        prev_count,
                        prev_count - 1,
                    )
                )
            return state_diffs
        elif ai_action.action.action_type == MbagAction.BREAK_BLOCK:
            block_id = malmo_state.blocks.blocks[ai_action.action.block_location]
            prev_count = block_counts[block_id]
            state_diffs = [
                BlockDiff(
                    ai_action.action.block_location,
                    block_id,
                    MinecraftBlocks.AIR,
                ),
            ]
            if not env_config["abilities"]["inf_blocks"]:
                state_diffs.append(
                    InventoryDiff(
                        player_index,
                        block_id,
                        prev_count,
                        prev_count + 1,
                    )
                )
            return state_diffs
        else:
            raise ValueError(
                f"unknown place/break action type {ai_action.action.action_type}"
            )
    elif isinstance(ai_action, MalmoMoveAIAction):
        return [
            LocationDiff(
                player_index,
                malmo_state.player_locations[player_index],
                ai_action.new_location,
            )
        ]
    elif isinstance(ai_action, MalmoGiveAIAction):
        giver_counts = get_block_counts_in_inventory(
            malmo_state.player_inventories[player_index]
        )
        recipient_counts = get_block_counts_in_inventory(
            malmo_state.player_inventories[ai_action.recipient_player_index]
        )
        block_id = ai_action.action.block_id

        return [
            InventoryDiff(
                player_index,
                block_id,
                giver_counts[block_id],
                giver_counts[block_id] - ai_action.blocks_to_give,
            ),
            InventoryDiff(
                ai_action.recipient_player_index,
                ai_action.action.block_id,
                recipient_counts[ai_action.action.block_id],
                recipient_counts[ai_action.action.block_id] + ai_action.blocks_to_give,
            ),
        ]
    else:
        raise ValueError(f"unknown AI action type {ai_action}")
