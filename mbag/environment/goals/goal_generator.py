from abc import ABC, abstractmethod
from typing import Any

from ..types import WorldSize
from ..blocks import MinecraftBlocks


class GoalGenerator(ABC):
    config: Any

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        ...
