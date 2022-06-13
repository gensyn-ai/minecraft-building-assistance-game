from typing import Dict, Type

from .goal_generator import GoalGenerator
from .grabcraft import CroppedGrabcraftGoalGenerator, GrabcraftGoalGenerator, SingleWallGrabcraftGenerator
from .simple import BasicGoalGenerator, RandomGoalGenerator
from .craftassist import CraftAssistGoalGenerator

ALL_GOAL_GENERATORS: Dict[str, Type[GoalGenerator]] = {
    "basic": BasicGoalGenerator,
    "random": RandomGoalGenerator,
    "grabcraft": GrabcraftGoalGenerator,
    "cropped_grabcraft": CroppedGrabcraftGoalGenerator,
    "single_wall_grabcraft": SingleWallGrabcraftGenerator,
    "craftassist": CraftAssistGoalGenerator,
}

__all__ = [
    "ALL_GOAL_GENERATORS",
    "BasicGoalGenerator",
    "RandomGoalGenerator",
    "GrabcraftGoalGenerator",
    "CroppedGrabcraftGoalGenerator",
    "SingleWallGrabcraftGenerator",
    "CraftAssistGoalGenerator",
]
