from typing import TypedDict, List

from .goal_generator import GoalGenerator
from ..blocks import MinecraftBlocks
from ..types import WorldSize


class GoalTransform(GoalGenerator):
    """
    GoalTransforms are filters or transformations that can be applied to generated
    goals. A GoalTransform can reject certain goals, alter a goal, and so on. It
    takes as input a GoalGenerator and should call this to get input goals; then, it
    modify them or ask for more goals if necessary.
    """

    def __init__(self, config: dict, goal_generator: GoalGenerator):
        super().__init__(config)
        self.goal_generator = goal_generator

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        return self.goal_generator.generate_goal(size)


class GoalTransformSpec(TypedDict, total=False):
    transform: str
    config: dict


class TransformedGoalGeneratorConfig(TypedDict):
    goal_generator: str
    goal_generator_config: dict
    goal_transforms: List[GoalTransformSpec]


class TransformedGoalGenerator(GoalGenerator):
    """
    Starts with a base goal generator and then applies GoalTransforms to it.
    """

    config: TransformedGoalGeneratorConfig

    def __init__(self, config: dict):
        super().__init__(config)

        from . import ALL_GOAL_GENERATORS, ALL_GOAL_TRANSFORMS

        base_goal_generator_class = ALL_GOAL_GENERATORS[self.config["goal_generator"]]
        self.base_goal_generator = base_goal_generator_class(
            self.config["goal_generator_config"]
        )
        goal_generator = self.base_goal_generator
        self.goal_transforms: List[GoalTransform] = []
        for transform_spec in self.config["goal_transforms"]:
            transform_class = ALL_GOAL_TRANSFORMS[transform_spec["transform"]]
            goal_generator = transform_class(
                transform_spec.get("config", {}), goal_generator
            )
            self.goal_transforms.append(goal_generator)

        self.goal_generator = goal_generator

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        return self.goal_generator.generate_goal(size)
