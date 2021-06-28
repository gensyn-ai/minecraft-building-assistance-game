from typing import Dict, Type

from .goal_generator import GoalGenerator
from .grabcraft import GrabcraftGoalGenerator
from .simple import BasicGoalGenerator, RandomGoalGenerator

ALL_GOAL_GENERATORS: Dict[str, Type[GoalGenerator]] = {
    "basic": BasicGoalGenerator,
    "random": RandomGoalGenerator,
    "grabcraft": GrabcraftGoalGenerator,
}

__all__ = [
    "ALL_GOAL_GENERATORS",
    "GrabcraftGoalGenerator",
    "BasicGoalGenerator",
    "RandomGoalGenerator",
]
