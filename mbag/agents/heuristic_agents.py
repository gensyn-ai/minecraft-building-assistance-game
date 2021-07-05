"""
A collection of agents following simple heuristics.
"""

from typing import Dict, List, Tuple, Type
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
            block_location_id = np.ravel_multi_index(
                block_location, self.env_config["world_size"]
            )

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


ALL_HEURISTIC_AGENTS: Dict[str, Type[MbagAgent]] = {
    "layer_builder": LayerBuilderAgent,
}
