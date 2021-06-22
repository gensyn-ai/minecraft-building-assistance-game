from typing import List, Literal, Tuple, TypedDict, cast
import numpy as np
from gym import spaces
import time
import logging

from .blocks import MinecraftBlocks
from .types import (
    MbagAction,
    MbagActionId,
    MbagInfoDict,
    MbagObs,
    WorldSize,
    num_world_obs_channels,
)

logger = logging.getLogger(__name__)


class MbagConfigDict(TypedDict):
    num_players: int
    horizon: int
    world_size: WorldSize

    goal_visibility: List[bool]
    """
    List with one boolean for each player, indicating if the player can observe the
    goal.
    """

    use_malmo: bool
    """
    Whether to connect to a real Minecraft instance with Project Malmo.
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

        if self.config["use_malmo"]:
            from .malmo import MalmoClient

            self.malmo_client = MalmoClient()

    def reset(self) -> List[MbagObs]:
        self.timestep = 0

        self.current_blocks = MinecraftBlocks(self.config["world_size"])
        self.current_blocks.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["bedrock"]
        self.current_blocks.blocks[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]

        self.goal_blocks = self._generate_goal()

        if self.config["use_malmo"]:
            self.malmo_client.start_mission(self.config, self.goal_blocks)
            time.sleep(1)  # Wait a second for the environment to load.

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

        if self.config["use_malmo"]:
            time.sleep(self.malmo_client.ACTION_DELAY)
            self._update_state_from_malmo()

        obs = [
            self._get_player_obs(player_index)
            for player_index in range(self.config["num_players"])
        ]
        rewards = [reward] * self.config["num_players"]
        dones = [self._done()] * self.config["num_players"]

        if dones[0] and self.config["use_malmo"]:
            for player_index in range(self.config["num_players"]):
                self.malmo_client.send_command(player_index, "quit")

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

            if place_break_result is not None and self.config["use_malmo"]:
                player_location, click_location = place_break_result
                self.malmo_client.send_command(
                    player_index,
                    "tp " + " ".join(map(str, player_location)),
                )
                viewpoint = np.array(player_location)
                viewpoint[1] += 1.6
                delta = np.array(click_location) - viewpoint
                delta /= np.sqrt((delta ** 2).sum())
                yaw = np.rad2deg(np.arctan2(-delta[0], delta[2]))
                pitch = np.rad2deg(-np.arcsin(delta[1]))
                self.malmo_client.send_command(player_index, f"setYaw {yaw}")
                self.malmo_client.send_command(player_index, f"setPitch {pitch}")

                # Give time to teleport.
                time.sleep(0.1)
                if action.action_type == MbagAction.PLACE_BLOCK:
                    self.malmo_client.send_command(player_index, "use 1")
                else:
                    self.malmo_client.send_command(player_index, "attack 1")

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

    def _update_state_from_malmo(self):
        malmo_state = self.malmo_client.get_observation(0)
        if malmo_state is None:
            return

        malmo_blocks = MinecraftBlocks.from_malmo_grid(
            self.config["world_size"], malmo_state["world"]
        )
        malmo_goal = MinecraftBlocks.from_malmo_grid(
            self.config["world_size"], malmo_state["goal"]
        )

        for location in map(
            tuple, np.argwhere(malmo_blocks.blocks != self.current_blocks.blocks)
        ):
            logger.error(
                f"block discrepancy at {location}: "
                "expected "
                f"{MinecraftBlocks.ID2NAME[self.current_blocks.blocks[location]]} "
                f"but received "
                f"{MinecraftBlocks.ID2NAME[malmo_blocks.blocks[location]]} "
                "from Malmo"
            )
        for location in map(
            tuple, np.argwhere(malmo_goal.blocks != self.goal_blocks.blocks)
        ):
            logger.error(
                f"goal discrepancy at {location}: "
                "expected "
                f"{MinecraftBlocks.ID2NAME[self.goal_blocks.blocks[location]]} "
                f"but received {MinecraftBlocks.ID2NAME[malmo_goal.blocks[location]]} "
                "from Malmo"
            )

    def _done(self) -> bool:
        return (
            self.timestep >= self.config["horizon"]
            or self.current_blocks == self.goal_blocks
        )
