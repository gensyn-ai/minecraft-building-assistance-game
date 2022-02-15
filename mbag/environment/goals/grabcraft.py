import glob
import os
import json
import random
import logging
from typing import Dict, List, Optional, Tuple
from typing_extensions import TypedDict, Literal

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_generator import GoalGenerator

logger = logging.getLogger(__name__)


class GrabcraftGoalConfig(TypedDict):
    data_dir: str
    subset: Literal["train", "val", "test"]


class StructureMetadata(TypedDict):
    id: str
    title: str
    description: str
    category: str
    slug: str
    tags: List[str]
    url: str


class StructureBlock(TypedDict):
    x: str
    y: str
    z: str
    hex: str
    rgb: Tuple[int, int, int]
    name: str
    mat_id: str
    file: str
    transparent: bool
    opacity: float
    texture: str


StructureJson = Dict[str, Dict[str, Dict[str, StructureBlock]]]


class GrabcraftGoalGenerator(GoalGenerator):
    config: GrabcraftGoalConfig
    structure_metadata: Dict[str, StructureMetadata]
    block_map: Dict[str, Tuple[str, Optional[str]]]

    def __init__(self, config: dict):
        super().__init__(config)

        self.data_dir = os.path.join(self.config["data_dir"], self.config["subset"])

        self._load_block_map()
        self._load_metadata()

    def _load_block_map(self):
        block_map_fname = os.path.join(
            os.path.dirname(__file__), "grabcraft_block_map.json"
        )
        with open(block_map_fname, "r") as block_map_file:
            self.block_map = json.load(block_map_file)

    def _load_metadata(self):
        self.structure_metadata = {}
        for metadata_fname in glob.glob(os.path.join(self.data_dir, "*.metadata.json")):
            with open(metadata_fname, "r") as metadata_file:
                metadata = json.load(metadata_file)
            structure_id = metadata["id"]
            if not os.path.exists(os.path.join(self.data_dir, f"{structure_id}.json")):
                continue  # Structure file does not exist.

            self.structure_metadata[structure_id] = metadata

    def _get_structure_size(self, structure_json: StructureJson) -> WorldSize:
        width, height, depth = 0, 0, 0
        for y_str, y_layer in structure_json.items():
            y = int(y_str)
            if y > height:
                height = y
            for x_str, x_layer in y_layer.items():
                x = int(x_str)
                if x > width:
                    width = x
                for z_str, block in x_layer.items():
                    z = int(z_str)
                    if z > depth:
                        depth = z
        return width, height, depth

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        success = False
        while not success:
            success = True

            structure_id = random.choice(list(self.structure_metadata.keys()))
            structure, structure_size = self._get_structure(structure_id)

            # check if structure is valid and within size constraints
            if structure is None or (
                structure_size[0] > size[0]
                or structure_size[1] > size[1]
                or structure_size[2] > size[2]
            ):
                success = False
                continue

            # Randomly place structure within world.
            goal = GoalGenerator.randomly_place_structure(structure, size)

            # Add a layer of dirt at the bottom of the structure wherever there's still
            # air.
            bottom_layer = goal.blocks[:, 0, :]
            bottom_layer[bottom_layer == MinecraftBlocks.AIR] = MinecraftBlocks.NAME2ID[
                "dirt"
            ]

        return goal

    def _get_structure(self, structure_id):
        with open(
            os.path.join(self.data_dir, f"{structure_id}.json"), "r"
        ) as structure_file:
            structure_json: StructureJson = json.load(structure_file)

        structure_size = self._get_structure_size(structure_json)
        structure = MinecraftBlocks(structure_size)
        structure.blocks[:] = MinecraftBlocks.AIR
        structure.block_states[:] = 0
        for y_str, y_layer in structure_json.items():
            y = int(y_str)
            for x_str, x_layer in y_layer.items():
                x = int(x_str)
                for z_str, block in x_layer.items():
                    z = int(z_str)
                    block_variant = self.block_map.get(block["name"])
                    if block_variant is None:
                        logger.warning(f"no map entry for \"{block['name']}\"")
                    else:
                        block_name, variant_name = block_variant
                        block_id = MinecraftBlocks.NAME2ID.get(block_name)
                        if block_id is not None:
                            structure.blocks[
                                x - 1,
                                y - 1,
                                z - 1,
                            ] = block_id
                        else:
                            return None, 0

        metadata = self.structure_metadata[structure_id]
        logger.info(f"chose structure {structure_id} ({metadata['title']})")

        return structure, structure_size


class CroppedGrabcraftGoalGenerator(GrabcraftGoalGenerator):
    @staticmethod
    def _fill_crop(
        initial_struct: MinecraftBlocks,
        crop: MinecraftBlocks,
        coords: Tuple[int, int, int],
    ) -> None:
        x, y, z = coords
        for i in range(len(crop.blocks)):
            if x + i >= len(initial_struct.blocks):
                break

            for j in range(len(crop.blocks[0])):
                if y + j >= len(initial_struct.blocks[0]):
                    break

                for k in range(len(crop.blocks[0][0])):
                    if z + k >= len(initial_struct.blocks[0][0]):
                        break

                    crop.blocks[i][j][k] = initial_struct.blocks[x + i][y + j][z + k]

    def generate_crops(
        self,
        size: WorldSize,
        restrict_crops: int = None,
        floating: bool = False,
        only_bottom_layer: bool = False,
    ) -> List[MinecraftBlocks]:
        success = False
        chosen_crops = []
        while not success:
            success = True

            structure_id = random.choice(list(self.structure_metadata.keys()))
            structure, structure_size = self._get_structure(structure_id)

            if structure is None:
                success = False
                continue

            valid_crops = []
            density_threshold = 0.1
            struct_density = structure.density()

            step_x = size[0] // 2
            step_y = size[1] // 2
            step_z = size[2] // 2

            crop_size = (
                min(size[0], structure_size[0]),
                min(size[1], structure_size[1]),
                min(size[2], structure_size[2]),
            )

            valid_y_range = len(structure.blocks[0])
            if only_bottom_layer:
                valid_y_range = 1

            for x in range(0, len(structure.blocks), step_x):
                for y in range(0, valid_y_range, step_y):
                    for z in range(0, len(structure.blocks[0][0]), step_z):
                        crop = MinecraftBlocks(crop_size)
                        crop.blocks[:] = MinecraftBlocks.AIR
                        crop.block_states[:] = 0

                        self._fill_crop(structure, crop, (x, y, z))

                        if (
                            abs(crop.density() - struct_density) / struct_density
                            <= density_threshold
                        ):
                            valid_crops.append(crop)

            if not floating:
                for i, crop in enumerate(valid_crops):
                    if not crop.is_continuous():
                        valid_crops.pop(i)

            if len(valid_crops) == 0:
                success = False
                continue

            chosen_crops = valid_crops
            if restrict_crops is not None:
                chosen_crops = [
                    chosen_crops.pop(random.randrange(len(chosen_crops)))
                    for _ in range(min(len(valid_crops), restrict_crops))
                ]

        return chosen_crops

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        crops = self.generate_crops(
            size, restrict_crops=10, floating=True, only_bottom_layer=False
        )

        rand_crop = random.choice(crops)
        # Randomly place structure within world.
        goal = GoalGenerator.randomly_place_structure(rand_crop, size)

        # Add a layer of dirt at the bottom of the structure wherever there's still
        # air.
        bottom_layer = goal.blocks[:, 0, :]
        bottom_layer[bottom_layer == MinecraftBlocks.AIR] = MinecraftBlocks.NAME2ID[
            "dirt"
        ]

        return goal
