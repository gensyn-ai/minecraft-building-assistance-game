from abc import ABC, abstractmethod
from typing import Any
import random

from ..types import WorldSize
from ..blocks import MinecraftBlocks


class GoalGenerator(ABC):
    config: Any

    def __init__(self, config: dict):
        self.config = config

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
