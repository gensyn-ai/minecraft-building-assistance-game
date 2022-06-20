import glob
import os
import json
import random
import logging
import numpy as np
import itertools
from typing import Dict, List, Optional, Tuple
from typing_extensions import TypedDict

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_generator import GoalGenerator

logger = logging.getLogger(__name__)


class CraftAssistGoalConfig(TypedDict):
    data_dir: str
    train: bool


class CraftAssistStats(TypedDict):
    size: WorldSize
    placed: int
    broken: int
    net_placed: int
    player_minutes: Dict[str, float]


class CraftAssistGoalGenerator(GoalGenerator):
    config: CraftAssistGoalConfig
    house_ids: List[str]
    block_map: Dict[str, Optional[Tuple[str, Optional[str]]]]

    def __init__(self, config: dict):
        super().__init__(config)
        self._load_block_map()
        self._load_house_ids()

    def _load_block_map(self):
        block_map_fname = os.path.join(
            os.path.dirname(__file__), "craftassist_block_map.json"
        )
        with open(block_map_fname, "r") as block_map_file:
            self.block_map = json.load(block_map_file)

    def _load_house_ids(self):
        self.house_ids = []
        print(
            os.path.join(
                self.config["data_dir"],
                "houses",
                "train" if self.config["train"] else "test",
                "*",
            )
        )
        for house_dir in glob.glob(
            os.path.join(
                self.config["data_dir"],
                "houses",
                "train" if self.config["train"] else "test",
                "*",
            )
        ):
            house_id = os.path.split(house_dir)[-1]
            self.house_ids.append(house_id)

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        success = False
        while not success:
            success = True

            house_id = random.choice(self.house_ids)
            house_data = np.load(
                os.path.join(
                    self.config["data_dir"],
                    "houses",
                    "train" if self.config["train"] else "test",
                    house_id,
                    "schematic.npy",
                ),
                "r",
            )

            # First, check if structure is too big.
            structure_size = house_data.shape[:3]
            if (
                structure_size[0] > size[0]
                or structure_size[1] > size[1]
                or structure_size[2] > size[2]
            ):
                success = False
                continue

            # Next, make sure all blocks are valid.
            structure = MinecraftBlocks(structure_size)
            structure.blocks[:] = MinecraftBlocks.AIR
            structure.block_states[:] = 0
            for x, y, z in itertools.product(
                range(structure_size[0]),
                range(structure_size[1]),
                range(structure_size[2]),
            ):
                minecraft_id, minecraft_data = house_data[x, y, z]
                minecraft_combined_id = f"{minecraft_id}:{minecraft_data}"
                try:
                    block_variant = self.block_map[minecraft_combined_id]
                except KeyError:
                    logger.warning(f"no map entry for {minecraft_combined_id}")
                    success = False
                else:
                    if block_variant is None:
                        block_name, variant_name = "air", None
                    else:
                        block_name, variant_name = block_variant
                    block_id = MinecraftBlocks.NAME2ID.get(block_name)
                    if block_id is not None:
                        structure.blocks[x, y, z] = block_id
                    else:
                        success = False

            # Randomly place structure within world.
            goal = GoalGenerator.randomly_place_structure(structure, size)

            # Add a layer of dirt at the bottom of the structure wherever there's still
            # air.
            goal = GoalGenerator.add_grass(goal)

        logger.info(f"chose house {house_id}")

        return goal
