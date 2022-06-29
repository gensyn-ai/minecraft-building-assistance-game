from abc import ABC, abstractmethod
from typing import Any
import random

from ..types import WorldSize
from ..blocks import MinecraftBlocks


class GoalGenerator(ABC):
    default_config: Any = {}
    config: Any

    def __init__(self, config: dict):
        self.config = dict(self.default_config)
        self.config.update(config)

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

    @staticmethod
    def add_grass(
        structure: MinecraftBlocks,
    ) -> MinecraftBlocks:
        """
        Add grass to the bottom layer of the structure and returns the result.
        """
        structure = structure.copy()

        bottom_layer = structure.blocks[:, 0, :]
        bottom_layer[
            bottom_layer == MinecraftBlocks.NAME2ID["air"]
        ] = MinecraftBlocks.NAME2ID["grass"]

        return structure

    @staticmethod
    def make_uniform(
        structure: MinecraftBlocks,
        block_type: int,
    ) -> MinecraftBlocks:
        """
        Modify structure so that every non-air block is the same specified block type, with an exception made to grass
        on the bottom layer, since that will be there at the start already.
        """
        new_structure = structure.copy()

        top_layers = new_structure.blocks[:, 1:, :]
        top_layers[top_layers != MinecraftBlocks.AIR] = block_type

        bottom_layer = new_structure.blocks[:, 0, :]
        bottom_layer[
            (bottom_layer != MinecraftBlocks.AIR)
            & (bottom_layer != MinecraftBlocks.NAME2ID["grass"])
        ] = block_type

        return new_structure
