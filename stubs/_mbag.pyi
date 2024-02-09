from typing import List, Optional

import numpy as np

from mbag.environment.types import BlockLocation, WorldLocation

def get_action_distribution_mask(
    world_obs: np.ndarray,
    inventory_obs: np.ndarray,
    timestep: int,
    teleportation: bool,
    inf_blocks: bool,
) -> np.ndarray:
    pass

def get_viewpoint_click_candidates(
    blocks: np.ndarray,
    action_type: int,
    block_location: BlockLocation,
    player_location: Optional[WorldLocation],
    other_player_locations: List[WorldLocation],
) -> np.ndarray:
    pass
