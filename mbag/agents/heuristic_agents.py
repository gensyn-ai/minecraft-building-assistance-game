"""
A collection of agents following simple heuristics.
"""

from typing import Dict, List, Tuple, Type
from queue import PriorityQueue
import numpy as np
import random

from ..environment.types import (
    BlockLocation,
    MbagActionTuple,
    MbagActionType,
    MbagObs,
    MbagAction,
)
from ..environment.blocks import MinecraftBlocks
from .mbag_agent import MbagAgent


class NoopAgent(MbagAgent):
    def get_action_type_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.zeros(MbagAction.NUM_ACTION_TYPES)
        action_dist[MbagAction.NOOP] = 1
        return action_dist


class HardcodedBuilderAgent(MbagAgent):
    """
    Builds the simple goal generator using a hardcoded predetermined sequence of moves
    """

    current_command: int

    def reset(self):
        self.current_command = 0

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        COMMAND_LIST: list = [
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
            (MbagAction.MOVE_POS_Y, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (0, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
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
            (MbagAction.MOVE_POS_Z, 0, 0),
        ]
        if self.current_command >= len(COMMAND_LIST):
            return (MbagAction.NOOP, 0, 0)

        action = COMMAND_LIST[self.current_command]
        self.current_command += 1
        return action

    def get_state(self) -> List[np.ndarray]:
        return [np.array([self.current_command])]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.current_command = int(state[0][0])


class MovementAgent(MbagAgent):
    """
    Moves around randomly
    """

    def get_action_type_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.zeros(MbagAction.NUM_ACTION_TYPES)
        action_dist[MbagAction.MOVE_POS_X] = 1 / 6
        action_dist[MbagAction.MOVE_NEG_X] = 1 / 6
        action_dist[MbagAction.MOVE_POS_Y] = 1 / 6
        action_dist[MbagAction.MOVE_NEG_Y] = 1 / 6
        action_dist[MbagAction.MOVE_POS_Z] = 1 / 6
        action_dist[MbagAction.MOVE_NEG_Z] = 1 / 6

        return action_dist


class LayerBuilderAgent(MbagAgent):
    """
    Builds the goal structure one layer at a time, from bottom to top.
    """

    current_layer: int

    def reset(self):
        self.current_layer = 0

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        (world_obs, _) = obs

        # Check if current layer is done.
        while self.current_layer < self.env_config["world_size"][1] and np.all(
            world_obs[:2, :, self.current_layer]
            == world_obs[2:4, :, self.current_layer]
        ):
            self.current_layer += 1

        action_type: MbagActionType
        if self.current_layer == self.env_config["world_size"][1]:
            action_type = MbagAction.NOOP
            return action_type, 0, 0
        else:
            layer_blocks = world_obs[0, :, self.current_layer, :]
            goal_blocks = world_obs[2, :, self.current_layer, :]

            layer_block_location: Tuple[int, int] = tuple(
                random.choice(np.argwhere(layer_blocks != goal_blocks))  # type: ignore
            )
            block_location: BlockLocation = (
                layer_block_location[0],
                self.current_layer,
                layer_block_location[1],
            )
            block_location_id = int(
                np.ravel_multi_index(block_location, self.env_config["world_size"])
            )

            block_id: int
            if layer_blocks[layer_block_location] == MinecraftBlocks.AIR:
                action_type = MbagAction.PLACE_BLOCK
                block_id = goal_blocks[layer_block_location]
            else:
                action_type = MbagAction.BREAK_BLOCK
                block_id = 0

            return action_type, block_location_id, block_id

    def get_state(self) -> List[np.ndarray]:
        return [np.array([self.current_layer])]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.current_layer = int(state[0][0])


class PriorityQueueAgent(MbagAgent):
    """
    Places the block with lowest layer that is reachable
    Assumes that there is a block at layer 0, otherwise the structure is floating
    Todo: Preprocess the goal?
    """

    seeding: bool  # Whether blocks have been placed yet
    blockFrontier: PriorityQueue  # PQ to store blocks and their layers

    def reset(self):
        self.blockFrontier = PriorityQueue()
        self.seeding = False

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        (world_obs, _) = obs

        # Check if we need to seed the PQ with a random initial block from the base layer
        if not self.seeding:
            self.seeding = True

            layer = 0
            while layer < self.env_config["world_size"][1] and np.all(
                world_obs[:2, :, layer] == world_obs[2:4, :, layer]
            ):
                layer += 1

            goal_blocks = world_obs[2, :, layer, :]
            layer_blocks = world_obs[0, :, layer, :]
            layer_block_location: Tuple[int, int] = tuple(
                random.choice(np.argwhere(layer_blocks != goal_blocks))  # type: ignore
            )
            self.blockFrontier.put((0, layer_block_location))

        print(self.blockFrontier)
        action_type: MbagActionType
        if self.blockFrontier.empty():
            action_type = MbagAction.NOOP
            return action_type, 0, 0
        else:
            layer, location = self.blockFrontier.get()

            axes = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
            for direction in axes:
                x = location[0] + direction[0]
                y = layer + direction[1]
                z = location[1] + direction[2]

                if (
                    x < 0
                    or y < 0
                    or z < 0
                    or x >= world_obs.shape[1]
                    or y >= world_obs.shape[2]
                    or z >= world_obs.shape[3]
                ):
                    continue

                goal_block = world_obs[2, x, y, z]
                actual_block = world_obs[0, x, y, z]
                if (
                    goal_block != actual_block
                    and not (y, (x, z)) in self.blockFrontier.queue
                ):
                    self.blockFrontier.put((y, (x, z)))

            if world_obs[0, location[0], layer, location[1]] == MinecraftBlocks.AIR:
                action_type = MbagAction.PLACE_BLOCK
                block_id = world_obs[2, location[0], layer, location[1]]
            else:
                action_type = MbagAction.BREAK_BLOCK
                block_id = 0

            block_location: BlockLocation = (
                location[0],
                layer,
                location[1],
            )
            block_location_id = int(
                np.ravel_multi_index(block_location, self.env_config["world_size"])
            )

            return action_type, block_location_id, block_id

    def get_state(self) -> List[np.ndarray]:
        return [np.array([self.seeding]), np.array([self.blockFrontier])]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.seeding = bool(state[0][0])
        self.blockFrontier = state[1][0]


ALL_HEURISTIC_AGENTS: Dict[str, Type[MbagAgent]] = {
    "layer_builder": LayerBuilderAgent,
    "priority_queue": PriorityQueueAgent,
}
