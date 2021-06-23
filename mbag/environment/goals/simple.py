from ..types import WorldSize
from ..blocks import MinecraftBlocks
from .goal_generator import GoalGenerator


class BasicGoalGenerator(GoalGenerator):
    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = MinecraftBlocks(size)
        goal.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["cobblestone"]
        return goal
