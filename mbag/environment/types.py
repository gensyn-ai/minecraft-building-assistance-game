from typing import Literal, Tuple, TypedDict, cast
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


class MbagInfoDict(TypedDict):
    pass
