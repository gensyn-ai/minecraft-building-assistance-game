from typing import List, TypedDict

import numpy as np

from .blocks import MinecraftBlocks
from .types import FacingDirection, MbagInventory, WorldLocation


class MbagStateDict(TypedDict):
    current_blocks: MinecraftBlocks
    goal_blocks: MinecraftBlocks
    player_locations: List[WorldLocation]
    player_directions: List[FacingDirection]
    player_inventories: List[MbagInventory]
    last_interacted: np.ndarray
    timestep: int
