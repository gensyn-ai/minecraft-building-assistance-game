from typing import List, Literal, Tuple, TypedDict, cast
import numpy as np
from gym import spaces

from .blocks import MinecraftBlocks
from .types import (
    MbagAction,
    MbagActionId,
    MbagInfoDict,
    MbagObs,
    WorldSize,
    num_world_obs_channels,
)


class MbagConfigDict(TypedDict):
    num_players: int
    horizon: int
    world_size: WorldSize

    goal_visibility: List[bool]
    """
    List with one boolean for each player, indicating if the player can observe the
    goal.
    """


class MbagEnv(object):

    current_blocks: MinecraftBlocks
    goal_blocks: MinecraftBlocks
    timestep: int

    def __init__(self, config: MbagConfigDict):
        self.config = config

        self.world_obs_shape = (num_world_obs_channels,) + self.config["world_size"]
        self.observation_space = spaces.Tuple(
            (spaces.Box(0, 255, self.world_obs_shape),)
        )
        self.action_space = spaces.Discrete(
            MbagAction.get_num_actions(self.config["world_size"])
        )

    def reset(self) -> List[MbagObs]:
        self.timestep = 0

        self.current_blocks = MinecraftBlocks(self.config["world_size"])
        self.current_blocks.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["bedrock"]
        self.current_blocks.blocks[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]

        self.goal_blocks = self._generate_goal()

        return [
            self._get_player_obs(player_index)
            for player_index in range(self.config["num_players"])
        ]

    def step(
        self, action_ids: List[MbagActionId]
    ) -> Tuple[List[MbagObs], List[float], List[bool], List[MbagInfoDict]]:
        assert len(action_ids) == self.config["num_players"], "Wrong number of actions."

        reward: float = 0
        infos: List[MbagInfoDict] = []

        for player_index, player_action_id in enumerate(action_ids):
            player_reward, player_info = self._step_player(
                player_index, player_action_id
            )
            reward += player_reward
            infos.append(player_info)

        self.timestep += 1

        obs = [
            self._get_player_obs(player_index)
            for player_index in range(self.config["num_players"])
        ]
        rewards = [reward] * self.config["num_players"]
        dones = [self._done()] * self.config["num_players"]

        return obs, rewards, dones, infos

    def _generate_goal(self) -> MinecraftBlocks:
        # TODO: generate more interesting goals
        goal = self.current_blocks.copy()
        goal.blocks[:, 2, :] = MinecraftBlocks.NAME2ID["cobblestone"]
        return goal

    def _step_player(
        self, player_index: int, action_id: MbagActionId
    ) -> Tuple[float, MbagInfoDict]:
        action = MbagAction(action_id, self.config["world_size"])

        reward: float = 0
        info: MbagInfoDict = {}

        if action.action_type == MbagAction.NOOP:
            pass
        elif action.action_type in [MbagAction.PLACE_BLOCK, MbagAction.BREAK_BLOCK]:
            prev_block = self.current_blocks[action.block_location]
            goal_block = self.goal_blocks[action.block_location]

            # Try to place or break block.
            place_break_result = self.current_blocks.try_break_place(
                cast(Literal[1, 2], action.action_type), action.block_location
            )

            if place_break_result is not None:
                pass  # TODO: place via Malmo if connected

            # Calculate reward based on progress towards goal.
            new_block = self.current_blocks[action.block_location]
            if new_block == goal_block and prev_block != goal_block:
                reward = 1
            elif new_block != goal_block and prev_block == goal_block:
                reward = -1

        return reward, info

    def _get_player_obs(self, player_index: int) -> MbagObs:
        world_obs = np.zeros(self.world_obs_shape, np.uint8)
        world_obs[0] = self.current_blocks.blocks
        world_obs[1] = self.current_blocks.block_states

        if self.config["goal_visibility"][player_index]:
            world_obs[2] = self.goal_blocks.blocks
            world_obs[3] = self.goal_blocks.block_states

        return (world_obs,)

    def _done(self) -> bool:
        return (
            self.timestep >= self.config["horizon"]
            or self.current_blocks == self.goal_blocks
        )
