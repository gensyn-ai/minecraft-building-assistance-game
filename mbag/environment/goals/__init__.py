from typing import Dict, Type

from .goal_generator import GoalGenerator
from .grabcraft import CroppedGrabcraftGoalGenerator, GrabcraftGoalGenerator
from .simple import BasicGoalGenerator, RandomGoalGenerator
from .craftassist import CraftAssistGoalGenerator

ALL_GOAL_GENERATORS: Dict[str, Type[GoalGenerator]] = {
    "basic": BasicGoalGenerator,
    "random": RandomGoalGenerator,
    "grabcraft": GrabcraftGoalGenerator,
    "cropped_grabcraft": CroppedGrabcraftGoalGenerator,
    "craftassist": CraftAssistGoalGenerator,
}

__all__ = [
    "ALL_GOAL_GENERATORS",
    "BasicGoalGenerator",
    "RandomGoalGenerator",
    "GrabcraftGoalGenerator",
    "CroppedGrabcraftGoalGenerator",
    "CraftAssistGoalGenerator",
]
