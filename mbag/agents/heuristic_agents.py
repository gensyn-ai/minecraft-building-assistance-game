"""
A collection of agents following simple heuristics.
"""

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


class NoopAgent(MbagAgent):
    def get_action_type_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.zeros(MbagAction.NUM_ACTION_TYPES)
        action_dist[MbagAction.NOOP] = 1
        return action_dist


class LayerBuilderAgent(MbagAgent):
    """
    Builds the goal structure one layer at a time, from bottom to top.
    """

    current_layer: int

    def reset(self):
        self.current_layer = 0

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        (world_obs,) = obs

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
    block_frontier: list  # PQ to store blocks and their layers

    def reset(self):
        self.block_frontier = []
        self.seeding = False

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        (world_obs,) = obs

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
            heapq.heappush(self.block_frontier, (0, layer_block_location))

        action_type: MbagActionType
        if len(self.block_frontier) == 0:
            action_type = MbagAction.NOOP
            return action_type, 0, 0
        else:
            layer, location = heapq.heappop(self.block_frontier)

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
                    and not (y, (x, z)) in self.block_frontier
                ):
                    heapq.heappush(self.block_frontier, (y, (x, z)))

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
        # print("Getting state")
        # print([np.array([self.seeding, self.block_frontier])])
        return [np.array([self.seeding, self.block_frontier], dtype=object)]

    def set_state(self, state: List[np.ndarray]) -> None:
        # print("setting state")
        # print(state)
        # print(self.block_frontier)
        self.seeding = bool(state[0][0])
        self.block_frontier = list(map(tuple, state[0][1]))


ALL_HEURISTIC_AGENTS: Dict[str, Type[MbagAgent]] = {
    "layer_builder": LayerBuilderAgent,
    "priority_queue": PriorityQueueAgent,
}
