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
    force_single_cc: bool


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
    default_config: GrabcraftGoalConfig = {
        "data_dir": "data/grabcraft",
        "subset": "train",
        "force_single_cc": False,
    }

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
            structure = self._get_structure(structure_id)

            # check if structure is valid and within size constraints
            if structure is None or (
                structure.size[0] > size[0]
                or structure.size[1] > size[1]
                or structure.size[2] > size[2]
            ):
                success = False
                continue

            # Randomly place structure within world.
            goal = GoalGenerator.randomly_place_structure(structure, size)

            # If we want to force the structure to be a single connected component,
            # then check here.
            if self.config["force_single_cc"]:
                if not goal.is_single_cc():
                    success = False

            # Add a layer of dirt at the bottom of the structure wherever there's still
            # air.
            bottom_layer = goal.blocks[:, 0, :]
            bottom_layer[bottom_layer == MinecraftBlocks.AIR] = MinecraftBlocks.NAME2ID[
                "dirt"
            ]

        return goal

    def _get_structure(self, structure_id: str) -> Optional[MinecraftBlocks]:
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
                            return None

        metadata = self.structure_metadata[structure_id]
        logger.info(f"chose structure {structure_id} ({metadata['title']})")

        return structure


class CroppedGrabcraftGoalConfig(GrabcraftGoalConfig):
    tethered_to_ground: bool
    density_threshold: float


class CroppedGrabcraftGoalGenerator(GrabcraftGoalGenerator):
    default_config: CroppedGrabcraftGoalConfig = {
        "data_dir": GrabcraftGoalGenerator.default_config["data_dir"],
        "subset": GrabcraftGoalGenerator.default_config["subset"],
        "force_single_cc": GrabcraftGoalGenerator.default_config["force_single_cc"],
        "tethered_to_ground": True,
        "density_threshold": 0.25,
    }

    config: CroppedGrabcraftGoalConfig

    def _generate_crop(self, size: WorldSize, retries: int = 5) -> MinecraftBlocks:
        while True:
            structure_id = random.choice(list(self.structure_metadata.keys()))
            structure = self._get_structure(structure_id)
            if structure is None:
                continue
            struct_density = structure.density()

            crop_size = (
                min(size[0], structure.size[0]),
                min(size[1], structure.size[1]),
                min(size[2], structure.size[2]),
            )

            x_range = structure.size[0] - 1
            y_range = 0 if self.config["tethered_to_ground"] else structure.size[1] - 1
            z_range = structure.size[2] - 1

            for retry_index in range(retries):
                rand_crop = MinecraftBlocks(crop_size)
                rand_crop.blocks[:] = MinecraftBlocks.AIR
                rand_crop.block_states[:] = 0
                rand_crop.fill_from_crop(
                    structure,
                    (
                        random.randint(0, x_range),
                        random.randint(0, y_range),
                        random.randint(0, z_range),
                    ),
                )

                if (
                    abs(rand_crop.density() - struct_density) / struct_density
                    > self.config["density_threshold"]
                ):
                    continue

                if self.config["force_single_cc"] and not rand_crop.is_single_cc():
                    continue

                return rand_crop

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        crop = self._generate_crop(size)

        # Randomly place structure within world.
        goal = GoalGenerator.randomly_place_structure(crop, size)

        # Add a layer of dirt at the bottom of the structure wherever there's still
        # air.
        bottom_layer = goal.blocks[:, 0, :]
        bottom_layer[bottom_layer == MinecraftBlocks.AIR] = MinecraftBlocks.NAME2ID[
            "dirt"
        ]

        return goal
