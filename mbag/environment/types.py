from typing import List, Tuple, cast
from typing_extensions import Literal, TypedDict
import numpy as np


WorldSize = Tuple[int, int, int]

BlockLocation = Tuple[int, int, int]

WorldLocation = Tuple[float, float, float]


MbagWorldObsArray = np.ndarray
"""
The world part of the observation in the form of a 4d numpy array of uint8. The last
three dimensions are spatial and the first is channels, each of which represents
different information about the world. The channels are
 0: current blocks
 1: current block states
 2: goal blocks
 3: goal block states
"""

num_world_obs_channels = 4

MbagObs = Tuple[MbagWorldObsArray]


MbagActionType = Literal[0, 1, 2]
MbagActionTuple = Tuple[MbagActionType, int, int]
"""
An action tuple (action_type, block_location, block_id).
"""


class MbagAction(object):
    """
    An action in MBAG which may or may not operate on a particular block.
    """

    NOOP: MbagActionType = 0
    PLACE_BLOCK: MbagActionType = 1
    BREAK_BLOCK: MbagActionType = 2

    NUM_ACTION_TYPES = 3
    ACTION_TYPE_NAMES = {
        NOOP: "NOOP",
        PLACE_BLOCK: "PLACE_BLOCK",
        BREAK_BLOCK: "BREAK_BLOCK",
    }

    action_type: MbagActionType
    block_location: BlockLocation
    block_id: int

    # Which actions require which attributes:
    BLOCK_ID_ACTION_TYPES = [PLACE_BLOCK]
    BLOCK_LOCATION_ACTION_TYPES = [PLACE_BLOCK, BREAK_BLOCK]

    def __init__(self, action_tuple: MbagActionTuple, world_size: WorldSize):
        self.action_type, block_location_index, self.block_id = action_tuple
        self.block_location = cast(
            BlockLocation, tuple(np.unravel_index(block_location_index, world_size))
        )

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


class MbagInfoDict(TypedDict):
    goal_similarity: float
    """
    Number representing how similar the current blocks in the world are to the goal
    structure. Higher is more similar. This can be used as a truer "reward" than the
    potentially shaped reward given to the agent by the environment.
    """

    own_reward: float
    """
    The reward from this step which is due to the current player's direct actions.
    """

    own_reward_prop: float
    """
    The current proportion of this player's reward which is coming from their own
    direct actions, as opposed to other agents'.
    """
