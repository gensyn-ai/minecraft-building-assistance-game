from typing import Literal, Tuple, TypedDict
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


MbagActionId = int


class MbagAction(object):
    """
    An action in MBAG which may or may not operate on a particular block.
    """

    NOOP = 0
    PLACE_BLOCK = 1
    BREAK_BLOCK = 2

    NUM_ACTIONS = 3

    action_type: Literal[0, 1, 2]
    block_location: BlockLocation

    def __init__(self, action_id: MbagActionId, world_size: WorldSize):
        action_parts = np.unravel_index(
            action_id, (MbagAction.NUM_ACTIONS,) + world_size
        )
        self.action_type = action_parts[0]
        self.block_location = action_parts[1:]

    @classmethod
    def get_action_shape(cls, world_size: WorldSize) -> Tuple[int, int, int, int]:
        return (cls.NUM_ACTIONS,) + world_size

    @classmethod
    def get_num_actions(cls, world_size: WorldSize) -> int:
        return cls.NUM_ACTIONS * int(np.prod(world_size))


class MbagInfoDict(TypedDict):
    pass
