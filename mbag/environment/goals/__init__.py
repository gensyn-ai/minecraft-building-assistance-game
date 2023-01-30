from typing import Dict, Type

from .craftassist import CraftAssistGoalGenerator
from .filters import DensityFilter, MinSizeFilter, SingleConnectedComponentFilter
from .goal_generator import GoalGenerator, GoalGeneratorConfig
from .goal_transform import GoalTransform, TransformedGoalGenerator
from .grabcraft import GrabcraftGoalGenerator
from .simple import (
    BasicGoalGenerator,
    RandomGoalGenerator,
    SetGoalGenerator,
    SimpleOverhangGoalGenerator,
)
from .transforms import (
    AddGrassTransform,
    AreaSampleTransform,
    CropTransform,
    MirrorTransform,
    RandomlyPlaceTransform,
    SeamCarvingTransform,
    UniformBlockTypeTransform,
)

ALL_GOAL_GENERATORS: Dict[str, Type[GoalGenerator]] = {
    "basic": BasicGoalGenerator,
    "random": RandomGoalGenerator,
    "simple_overhang": SimpleOverhangGoalGenerator,
    "grabcraft": GrabcraftGoalGenerator,
    "craftassist": CraftAssistGoalGenerator,
    "set_goal": SetGoalGenerator,
}

ALL_GOAL_TRANSFORMS: Dict[str, Type[GoalTransform]] = {
    "single_cc_filter": SingleConnectedComponentFilter,
    "density_filter": DensityFilter,
    "min_size_filter": MinSizeFilter,
    "randomly_place": RandomlyPlaceTransform,
    "add_grass": AddGrassTransform,
    "crop": CropTransform,
    "uniform_block_type": UniformBlockTypeTransform,
    "mirror": MirrorTransform,
    "area_sample": AreaSampleTransform,
    "seam_carve": SeamCarvingTransform,
}

__all__ = [
    "ALL_GOAL_GENERATORS",
    "ALL_GOAL_TRANSFORMS",
    "GoalGenerator",
    "GoalGeneratorConfig",
    "GoalTransform",
    "TransformedGoalGenerator",
    "BasicGoalGenerator",
    "RandomGoalGenerator",
    "GrabcraftGoalGenerator",
    "CraftAssistGoalGenerator",
]
