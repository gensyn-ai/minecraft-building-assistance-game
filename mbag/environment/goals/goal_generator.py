from abc import ABC, abstractmethod
from typing import Any
import random

from ..types import WorldSize
from ..blocks import MinecraftBlocks


class GoalGenerator(ABC):
    default_config: Any = {"pallette": False}
    config: Any

    def __init__(self, config: dict):
        self.config = dict(self.default_config)
        self.config.update(config)

    def generate_goal_scene(self, size: WorldSize) -> MinecraftBlocks:

        if not self.config["pallette"]:
            return self.generate_goal(size)

        print("Hello")
        goal = self.generate_goal((size[0] - 1, size[1], size[2]))
        goal_with_pallette = MinecraftBlocks(size)
        goal_with_pallette.block_states[: size[0] - 1] = goal.block_states
        goal_with_pallette.blocks[: size[0] - 1] = goal.blocks

        for index, block in enumerate(MinecraftBlocks.PLACEABLE_BLOCK_IDS):
            if index >= size[2]:
                break
            goal_with_pallette.blocks[size[0] - 1, 1, index] = block
            goal_with_pallette.block_states[size[0] - 1, 1, index] = 0
        return goal_with_pallette

    @abstractmethod
    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        ...

    @staticmethod
    def randomly_place_structure(
        structure: MinecraftBlocks,
        size: WorldSize,
    ) -> MinecraftBlocks:
        """
        Given a structure of size smaller than the given size, randomly places the
        structure along the x and z axes (but always keeps it at the same height on
        the y axis).
        """

        blocks = MinecraftBlocks(size)

        offset_x = random.randint(0, size[0] - structure.size[0])
        offset_z = random.randint(0, size[2] - structure.size[2])
        structure_slice = (
            slice(offset_x, offset_x + structure.size[0]),
            slice(0, structure.size[1]),
            slice(offset_z, offset_z + structure.size[2]),
        )
        blocks.blocks[structure_slice] = structure.blocks
        blocks.block_states[structure_slice] = structure.block_states

        return blocks
