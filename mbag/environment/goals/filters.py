"""
Various GoalTransforms which filter the possible goals.
"""

from typing import TypedDict

from ..types import WorldSize
from ..blocks import MinecraftBlocks
from .goal_transform import GoalTransform


class SingleConnectedComponentFilter(GoalTransform):
    """
    Filters out any structures which do not consist of a single connected component
    attached to the ground. Structures which pass this filter are able to be built in
    Minecraft without any scaffolding which makes construction easier.
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        is_single_cc = False
        while not is_single_cc:
            goal = super().generate_goal(size)
            is_single_cc = goal.is_single_cc()

        return goal


class DensityFilterConfig(TypedDict):
    min_density: float
    max_density: float


class DensityFilter(GoalTransform):
    """
    Filters structures with density outside of a specified range.
    """

    default_config: DensityFilterConfig = {
        "min_density": 0,
        "max_density": 1,
    }
    config: DensityFilterConfig

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        valid_density = False
        while not valid_density:
            goal = super().generate_goal(size)
            valid_density = (
                self.config["min_density"]
                <= goal.density()
                <= self.config["max_density"]
            )
        return goal
