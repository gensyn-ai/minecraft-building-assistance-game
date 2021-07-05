from typing import Set
from ..types import WorldSize
from ..blocks import MinecraftBlocks
from .goal_generator import GoalGenerator

import numpy as np
import cc3d


class BasicGoalGenerator(GoalGenerator):
    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = MinecraftBlocks(size)
        goal.blocks[:, :2, :] = MinecraftBlocks.NAME2ID["cobblestone"]
        return goal


class RandomGoalGenerator(GoalGenerator):
    """
    Generates a random structure of connected blocks.
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = MinecraftBlocks(size)
        structure_mask_noise = np.random.rand(*goal.blocks.shape) < 0.6
        structure_mask_ccs = cc3d.connected_components(
            structure_mask_noise, connectivity=6
        )
        ground_ccs: Set[int] = set(structure_mask_ccs[:, 0, :].reshape(-1).tolist())
        if np.any(~structure_mask_noise[:, 0, :]):
            ground_ccs.remove(0)
        structure_mask = np.isin(structure_mask_ccs, list(ground_ccs))

        block_ids = (
            np.random.randint(len(MinecraftBlocks.ID2NAME) - 2, size=goal.blocks.shape)
            + 2
        )

        goal.blocks[structure_mask] = block_ids[structure_mask]
        goal.blocks[:, 0, :][~structure_mask[:, 0, :]] = MinecraftBlocks.NAME2ID["dirt"]

        return goal
