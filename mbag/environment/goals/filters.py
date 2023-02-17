"""
Various GoalTransforms which filter the possible goals.
"""

import logging
from typing import TypedDict

from typing_extensions import Literal

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_transform import GoalTransform

logger = logging.getLogger(__name__)


class GoalFilter(GoalTransform):
    """
    A special case of a GoalTransform that just generates goals until one passes the
    filter(size, goal) method.
    """

    def filter(self, size: WorldSize, goal: MinecraftBlocks) -> bool:
        raise NotImplementedError()

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        success = False
        while not success:
            goal = super().generate_goal(size)
            success = self.filter(size, goal)
            if not success:
                logger.info(f"{self.__class__.__name__} rejected goal")
        return goal


class SingleConnectedComponentFilterConfig(TypedDict):
    connectivity: Literal[6, 18, 26]


class SingleConnectedComponentFilter(GoalFilter):
    """
    Filters out any structures which do not consist of a single connected component
    attached to the ground. Structures which pass this filter are able to be built in
    Minecraft without any scaffolding which makes construction easier.
    """

    default_config: SingleConnectedComponentFilterConfig = {"connectivity": 6}
    config: SingleConnectedComponentFilterConfig

    def filter(self, size: WorldSize, goal: MinecraftBlocks) -> bool:
        if goal.size == (1, 1, 1):
            return True
        return goal.is_single_cc(connectivity=self.config["connectivity"])


class DensityFilterConfig(TypedDict):
    min_density: float
    max_density: float


class DensityFilter(GoalFilter):
    """
    Filters structures with density outside of a specified range.
    """

    default_config: DensityFilterConfig = {
        "min_density": 0,
        "max_density": 1,
    }
    config: DensityFilterConfig

    def filter(self, size: WorldSize, goal: MinecraftBlocks) -> bool:
        return (
            self.config["min_density"] <= goal.density() <= self.config["max_density"]
        )


class MinSizeFilterConfig(TypedDict):
    min_size: WorldSize


class MinSizeFilter(GoalFilter):
    """
    Filters structures which are not at least the given size.
    """

    default_config: MinSizeFilterConfig = {
        "min_size": (1, 1, 1),
    }
    config: MinSizeFilterConfig

    def filter(self, size: WorldSize, goal: MinecraftBlocks) -> bool:
        goal_width, goal_height, goal_depth = goal.size
        min_width, min_height, min_depth = self.config["min_size"]
        return (
            goal_width >= min_width
            and goal_height >= min_height
            and goal_depth >= min_depth
        )
