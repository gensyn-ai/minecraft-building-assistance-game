from typing import Dict, List, Tuple, Type
import numpy as np
import random
import heapq

from ..environment.types import (
    BlockLocation,
    MbagActionTuple,
    MbagActionType,
    MbagObs,
    MbagAction,
)
from ..environment.blocks import MinecraftBlocks
from .mbag_agent import MbagAgent


class HardcodedBuilderAgent(MbagAgent):
    """
    Builds the simple goal generator using a hardcoded predetermined sequence of moves
    """

    current_command: int

    def reset(self):
        self.current_command = 0

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        command_list: List[MbagActionTuple] = [
            (MbagAction.MOVE_NEG_Y, 0, 0),
            (MbagAction.MOVE_POS_Y, 0, 0),
            (MbagAction.MOVE_NEG_X, 0, 0),
            (MbagAction.MOVE_NEG_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_X, 0, 0),
            (MbagAction.MOVE_POS_X, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_Y, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_Z, 0, 0),
            # (MbagAction.MOVE_NEG_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
        ]
        if self.current_command >= len(command_list):
            return (MbagAction.NOOP, 0, 0)

        action = command_list[self.current_command]
        self.current_command += 1
        return action

    def get_state(self) -> List[np.ndarray]:
        return [np.array([self.current_command])]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.current_command = int(state[0][0])


class HardcodedResourceAgent(MbagAgent):
    """
    Gets resources, then tries to build the simple goal generator
    """

    current_command: int

    def reset(self):
        self.current_command = 0

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        command_list: List[MbagActionTuple] = [
            (MbagAction.MOVE_POS_X, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (
                MbagAction.BREAK_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                0,
            ),
            (
                MbagAction.BREAK_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                0,
            ),
            (
                MbagAction.BREAK_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                0,
            ),
            (MbagAction.MOVE_NEG_Y, 0, 0),
            (MbagAction.MOVE_POS_Y, 0, 0),
            (MbagAction.MOVE_NEG_X, 0, 0),
            (MbagAction.MOVE_NEG_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_X, 0, 0),
            (MbagAction.MOVE_POS_X, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_Y, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_Z, 0, 0),
            # (MbagAction.MOVE_NEG_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
        ]
        if self.current_command >= len(command_list):
            return (MbagAction.NOOP, 0, 0)

        action = command_list[self.current_command]
        self.current_command += 1
        return action

    def get_state(self) -> List[np.ndarray]:
        return [np.array([self.current_command])]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.current_command = int(state[0][0])
