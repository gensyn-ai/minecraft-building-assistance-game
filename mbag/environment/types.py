from collections import namedtuple
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal, TypedDict

if TYPE_CHECKING:
    from .malmo import MalmoObservationDict


WorldSize = Tuple[int, int, int]

BlockLocation = Tuple[int, int, int]

WorldLocation = Tuple[float, float, float]

MbagInventoryObs = np.ndarray
"""
1D array mapping block ids with number held in inventory.
"""

FacingDirection = Tuple[float, float]  # Degrees horizontally, then vertically

MbagWorldObsArray = np.ndarray
"""
The world part of the observation in the form of a 4d numpy array of uint8. The last
three dimensions are spatial and the first is channels, each of which represents
different information about the world. The channels are
 0: current blocks
 1: current block states
 2: goal blocks
 3: goal block states
 4: player locations
 5: player who last interacted with block (if any)
"""

CURRENT_BLOCKS = 0
CURRENT_BLOCK_STATES = 1
GOAL_BLOCKS = 2
GOAL_BLOCK_STATES = 3
PLAYER_LOCATIONS = 4
LAST_INTERACTED = 5
num_world_obs_channels = 6

MbagObs = Tuple[MbagWorldObsArray, MbagInventoryObs, NDArray[np.int32]]
"""Tuple of (world_obs, inventory_obs, timestep)."""

MbagHumanCommandType = Literal["key", "mouse"]
MbagHumanCommand = Literal[
    "forward", "right", "left", "back", "attack", "inventory", "use"
]

MbagInventory = np.ndarray
"""
Player inventory will be stored as 2d numpy array.
Inventory slots are stored from 0 to 35 inclusive
First dimension is which inventory slot is being accessed
Second dimension is 0 for block id, 1 for block count
"""


MbagActionType = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MbagActionTuple = Tuple[MbagActionType, int, int]
"""
An action tuple (action_type, block_location, block_id).
"""
MBAG_ACTION_BREAK_PALETTE_NAME = "break_palette"


class MbagAction(object):
    """
    An action in MBAG which may or may not operate on a particular block.
    """

    NOOP: MbagActionType = 0
    PLACE_BLOCK: MbagActionType = 1
    BREAK_BLOCK: MbagActionType = 2
    MOVE_POS_X: MbagActionType = 3
    MOVE_NEG_X: MbagActionType = 4
    MOVE_POS_Y: MbagActionType = 5
    MOVE_NEG_Y: MbagActionType = 6
    MOVE_POS_Z: MbagActionType = 7
    MOVE_NEG_Z: MbagActionType = 8
    GIVE_BLOCK: MbagActionType = 9

    NUM_ACTION_TYPES = 10
    ACTION_TYPES: List[MbagActionType] = [
        NOOP,
        PLACE_BLOCK,
        BREAK_BLOCK,
        MOVE_POS_X,
        MOVE_NEG_X,
        MOVE_POS_Y,
        MOVE_NEG_Y,
        MOVE_POS_Z,
        MOVE_NEG_Z,
        GIVE_BLOCK,
    ]
    ACTION_TYPE_NAMES = {
        NOOP: "NOOP",
        PLACE_BLOCK: "PLACE_BLOCK",
        BREAK_BLOCK: "BREAK_BLOCK",
        MOVE_POS_X: "MOVE_POS_X",
        MOVE_NEG_X: "MOVE_NEG_X",
        MOVE_POS_Y: "MOVE_POS_Y",
        MOVE_NEG_Y: "MOVE_NEG_Y",
        MOVE_POS_Z: "MOVE_POS_Z",
        MOVE_NEG_Z: "MOVE_NEG_Z",
        GIVE_BLOCK: "GIVE_BLOCK",
    }

    action_type: MbagActionType
    block_location_index: int
    block_location: BlockLocation
    block_id: int

    # Which actions require which attributes:
    BLOCK_ID_ACTION_TYPES = [PLACE_BLOCK, GIVE_BLOCK]
    BLOCK_LOCATION_ACTION_TYPES = [PLACE_BLOCK, BREAK_BLOCK, GIVE_BLOCK]
    MOVE_ACTION_TYPES = [
        MOVE_POS_X,
        MOVE_NEG_X,
        MOVE_POS_Y,
        MOVE_NEG_Y,
        MOVE_POS_Z,
        MOVE_NEG_Z,
    ]

    MOVE_ACTION_MASK: Dict[MbagActionType, Tuple[WorldLocation, str]] = {
        MOVE_POS_X: ((1, 0, 0), "moveeast 1"),
        MOVE_NEG_X: ((-1, 0, 0), "movewest 1"),
        MOVE_POS_Y: ((0, 1, 0), "tp"),
        MOVE_NEG_Y: ((0, -1, 0), "tp"),
        MOVE_POS_Z: ((0, 0, 1), "movesouth 1"),
        MOVE_NEG_Z: ((0, 0, -1), "movenorth 1"),
    }

    def __init__(self, action_tuple: MbagActionTuple, world_size: WorldSize):
        self.action_type, self.block_location_index, self.block_id = action_tuple
        self.block_location = cast(
            BlockLocation,
            tuple(np.unravel_index(self.block_location_index, world_size)),
        )
        self._world_size = world_size

    def __str__(self):
        from .blocks import MinecraftBlocks

        parts: List[str] = [MbagAction.ACTION_TYPE_NAMES[self.action_type]]
        if self.action_type in MbagAction.BLOCK_ID_ACTION_TYPES:
            parts.append(MinecraftBlocks.ID2NAME[self.block_id])
        if self.action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES:
            parts.append(str(self.block_location))
        return " ".join(parts)

    def __repr__(self):
        return f"MbagAction<{self}>"

    def to_tuple(self) -> MbagActionTuple:
        return (self.action_type, self.block_location_index, self.block_id)

    def __eq__(self, other_action: object):
        if not isinstance(other_action, MbagAction):
            return False
        return self.to_tuple() == other_action.to_tuple()

    def to_json(self):
        print("Serializing JSON")
        return self.__str__()

    @classmethod
    def noop_action(cls):
        return cls((MbagAction.NOOP, 0, 0), (1, 1, 1))

    def is_palette(self, inf_blocks: bool) -> bool:
        """Returns whether this action is on the palette."""
        # The action can only be on the palette if inf_blocks is False,
        # otherwise the agent does not need to collect blocks and the palette
        # does not exist.
        return (self.block_location[0] == self._world_size[0] - 1) and not inf_blocks


class MbagPlaceBreakAIAction:
    def __init__(
        self,
        action: MbagAction,
        inventory_slot: int,
        yaw: float,
        pitch: float,
        player_location: WorldLocation,
    ):
        self.action = action
        self.inventory_slot = inventory_slot
        self.yaw = yaw
        self.pitch = pitch
        self.player_location = player_location


class MbagMoveAIAction:
    def __init__(self, action: MbagAction, player_location: WorldLocation):
        self.action = action
        self.player_location = player_location


class MbagGiveAIAction:
    def __init__(self, action: MbagAction, giver_index: int, receiver_index: int):
        self.action = action
        self.giver_index = giver_index
        self.receiver_index = receiver_index


MbagMalmoAIAction = Union[MbagPlaceBreakAIAction, MbagMoveAIAction, MbagGiveAIAction]


class BlockDiff(NamedTuple):
    location: BlockLocation
    old_block: int
    new_block: int


class InventoryDiff(NamedTuple):
    player_id: int
    block_id: int
    old_number: int
    new_number: int


class LocationDiff(NamedTuple):
    player_id: int
    old_location: WorldLocation
    new_location: WorldLocation


MalmoStateDiff = Union[BlockDiff, InventoryDiff, LocationDiff]


class MbagInfoDict(TypedDict):
    goal_similarity: float
    """
    Number representing how similar the current blocks in the world are to the goal
    structure. Higher is more similar. This can be used as a truer "reward" than the
    potentially shaped reward given to the agent by the environment.
    """

    goal_dependent_reward: float
    """
    The reward from this step which is due to the current player's actions and which
    depends on the goal.
    """

    goal_independent_reward: float
    """
    The reward from this step which is due to the current player's actions but which
    does not depend on the goal, i.e., bonuses or penalties for no-ops and actions,
    resource gathering bonuses, etc.
    """

    own_reward: float
    """
    The reward from this step which is due to the current player's direct actions, i.e.
    the sum of goal_dependent_reward and goal_independent_reward.
    """

    own_reward_prop: float
    """
    The current proportion of this player's reward which is coming from their own
    direct actions, as opposed to other agents'.
    """

    attempted_action: MbagAction
    """
    The action that the player tried to take.
    """

    action: MbagAction
    """
    The action that the player effectively took. That is, if the player attempted to
    do something but it didn't actually affect the world, it is logged as NOOP.
    """

    action_correct: bool
    """
    Whether an action directly contributed to the goal, either by placing the correct
    block or breakin an incorrect block.
    """

    malmo_observations: List[Tuple[datetime, "MalmoObservationDict"]]
    """
    If this player is a human agent, then this is the full timestamped list of
    observations from Malmo since the last timestep.
    """

    human_actions: List[MbagActionTuple]
    """
    If this player is a human agent, then this is a list of actions that have been
    deduced from what the human is doing in Malmo.
    """
