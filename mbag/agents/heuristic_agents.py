"""
A collection of agents following simple heuristics.
"""

from mbag.environment.blocks import MinecraftBlocks
import numpy as np

from ..environment.types import MbagObs, MbagAction
from .mbag_agent import MbagAgent


class NoopAgent(MbagAgent):
    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.zeros(
            MbagAction.get_action_shape(self.env_config["world_size"])
        )
        action_dist[MbagAction.NOOP] = 1
        action_dist /= action_dist.sum()
        return action_dist


class LayerBuilderAgent(MbagAgent):
    """
    Builds the goal structure one layer at a time, from bottom to top.
    """

    current_layer: int

    def reset(self):
        self.current_layer = 0

    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        (world_obs,) = obs
        action_dist = np.zeros(
            MbagAction.get_action_shape(self.env_config["world_size"])
        )

        # Check if current layer is done.
        while self.current_layer < self.env_config["world_size"][1] and np.all(
            world_obs[:2, :, self.current_layer]
            == world_obs[2:4, :, self.current_layer]
        ):
            self.current_layer += 1

        if self.current_layer == self.env_config["world_size"][1]:
            action_dist[MbagAction.NOOP] = 1
        else:
            layer_blocks = world_obs[0, :, self.current_layer, :]
            goal_blocks = world_obs[2, :, self.current_layer, :]
            action_dist[MbagAction.PLACE_BLOCK, :, self.current_layer, :] = (
                layer_blocks == MinecraftBlocks.AIR
            ) & (goal_blocks != MinecraftBlocks.AIR)
            action_dist[MbagAction.BREAK_BLOCK, :, self.current_layer, :] = (
                layer_blocks != MinecraftBlocks.AIR
            ) & (goal_blocks == MinecraftBlocks.AIR)

        action_dist /= action_dist.sum()
        return action_dist
