from typing import Dict, Type

from .goal_generator import GoalGenerator, GoalGeneratorConfig
from .goal_transform import GoalTransform, TransformedGoalGenerator
from .grabcraft import (
    GrabcraftGoalGenerator,
)
from .simple import BasicGoalGenerator, RandomGoalGenerator
from .craftassist import CraftAssistGoalGenerator
from .filters import SingleConnectedComponentFilter, DensityFilter, MinSizeFilter
from .transforms import (
    RandomlyPlaceTransform,
    AddGrassTransform,
    CropTransform,
    UniformBlockTypeTransform,
    MirrorTransform,
)

ALL_GOAL_GENERATORS: Dict[str, Type[GoalGenerator]] = {
    "basic": BasicGoalGenerator,
    "random": RandomGoalGenerator,
    "grabcraft": GrabcraftGoalGenerator,
    "craftassist": CraftAssistGoalGenerator,
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
