from typing import Dict, Optional, TypedDict, Union
import gym
import numpy as np
from gym import spaces
from ray.tune.registry import register_env
from ray.rllib.utils.typing import AgentID

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import MbagConfigDict, MbagStateDict
from mbag.environment.types import (
    CURRENT_BLOCK_STATES,
    CURRENT_BLOCKS,
    GOAL_BLOCK_STATES,
    GOAL_BLOCKS,
    MbagObs,
)
from .rllib_env import (
    FlatActionSpaceWrapper,
    MbagMultiAgentEnv,
    MbagRllibWrapper,
    unwrap_mbag_env,
)


class MbagObsWithMask(TypedDict):
    obs: MbagObs
    action_mask: np.ndarray


class MbagEnvModelStateDict(MbagStateDict, total=False):
    last_obs: Union[MbagObs, MbagObsWithMask]


class MbagEnvModel(gym.Env):
    """
    A single-agent environment model that can be used for planning in algorithms such
    as AlphaZero.
    """

    action_space: spaces.Discrete
    last_obs: Union[MbagObs, MbagObsWithMask]
    last_obs_dict: Dict[AgentID, MbagObs]

    def __init__(
        self,
        env: MbagRllibWrapper,
        config: MbagConfigDict,
        player_index: int = 0,
        include_action_mask_in_obs=True,
    ):
        super().__init__()

        self.env = env
        self.config = config
        self.set_player_index(player_index)
        self.agent_id = f"player_{player_index}"
        self.include_action_mask_in_obs = include_action_mask_in_obs

        assert isinstance(self.env.action_space, spaces.Discrete)
        self.action_space = self.env.action_space
        if include_action_mask_in_obs:
            self.observation_space = spaces.Dict(
                {
                    "obs": self.env.observation_space,
                    "action_mask": spaces.Box(
                        np.zeros(self.action_space.n, dtype=bool),
                        np.ones(self.action_space.n, dtype=bool),
                    ),
                }
            )
        else:
            self.observation_space = self.env.observation_space

    def set_player_index(self, player_index: int):
        self.player_index = player_index
        self.agent_id = f"player_{player_index}"

    def _store_last_obs_dict(self, obs_dict):
        self.last_obs_dict = obs_dict

    def _process_obs(self, obs: MbagObs):
        if self.include_action_mask_in_obs:
            world_obs, inventory_obs, timestep = obs
            self.last_obs = {
                "obs": obs,
                "action_mask": MbagActionDistribution.get_mask_flat(
                    self.config,
                    (world_obs[None], inventory_obs[None], timestep[None]),
                )[0],
            }
        else:
            self.last_obs = obs
        return self.last_obs

    def reset(self):
        obs_dict = self.env.reset()
        self._store_last_obs_dict(obs_dict)
        return self._process_obs(obs_dict[self.agent_id])

    def step(self, action):
        action_dict = {self.agent_id: action}
        for other_agent_id in self.last_obs_dict:
            if other_agent_id != self.agent_id:
                action_dict[other_agent_id] = 0  # NOOP for now

        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        self._store_last_obs_dict(obs_dict)
        return (
            self._process_obs(obs_dict[self.agent_id]),
            reward_dict[self.agent_id],
            done_dict.get(self.agent_id, done_dict["__all__"]),
            info_dict[self.agent_id],
        )

    def get_state(self) -> MbagEnvModelStateDict:
        env_state = self.env.get_state()
        return {
            "current_blocks": env_state["current_blocks"],
            "goal_blocks": env_state["goal_blocks"],
            "player_locations": env_state["player_locations"],
            "player_directions": env_state["player_directions"],
            "player_inventories": env_state["player_inventories"],
            "last_interacted": env_state["last_interacted"],
            "timestep": env_state["timestep"],
            "last_obs": self.last_obs,
        }

    def set_state(self, state: MbagEnvModelStateDict):
        if "last_obs" in state:
            self.env.set_state_no_obs(state)
            self.last_obs = state["last_obs"]
            # TODO: load last obs dict?
            return self.last_obs
        else:
            obs_dict = self.env.set_state(state)
            self._store_last_obs_dict(obs_dict)
            return self._process_obs(obs_dict[self.agent_id])

    def get_all_rewards(
        self, obs_batch: MbagObs, player_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Given a batch of observations, get the rewards for all possible actions
        as an array of shape (batch_size, NUM_CHANNELS, width, height, depth),
        where NUM_CHANNELS as defined as in MbagActionDistribution.

        The rewards returned here are only valid if the action is not a no-op.
        If the action is a noop (e.g., if the action is invalid) then the actual
        reward is 0.
        """

        if player_index is None:
            player_index = self.player_index

        world_obs, _, _ = obs_batch
        batch_size, _, width, height, depth = world_obs.shape
        env = unwrap_mbag_env(self)

        rewards = np.zeros(
            (batch_size, MbagActionDistribution.NUM_CHANNELS, width, height, depth)
        )

        # We only get reward for two actions: BREAK_BLOCK and PLACE_BLOCK.
        goal = (world_obs[:, GOAL_BLOCKS], world_obs[:, GOAL_BLOCK_STATES])
        similarity_before = env._get_goal_similarity(
            (world_obs[:, CURRENT_BLOCKS], world_obs[:, CURRENT_BLOCK_STATES]),
            goal,
            partial_credit=True,
            player_index=player_index,
        )

        block_states = np.zeros_like(world_obs[:, CURRENT_BLOCK_STATES])
        for block_id in range(MinecraftBlocks.NUM_BLOCKS):
            block_ids = np.full_like(world_obs[:, CURRENT_BLOCKS], block_id)
            similarity_after = env._get_goal_similarity(
                (block_ids, block_states),
                goal,
                partial_credit=True,
                player_index=player_index,
            )
            block_id_reward = similarity_after - similarity_before
            if block_id == MinecraftBlocks.AIR:
                rewards[:, MbagActionDistribution.BREAK_BLOCK] = block_id_reward
            rewards[:, MbagActionDistribution.PLACE_BLOCK][
                :, block_id
            ] = block_id_reward

        # TODO: implement shaping reward for gathering resources and lack of reward
        # for breaking palette blocks

        return rewards


def create_mbag_env_model(
    config: MbagConfigDict, player_index: int = 0, include_action_mask_in_obs=True
) -> MbagEnvModel:
    # We should never use Malmo in the env model.
    config["malmo"]["use_malmo"] = False
    env = MbagMultiAgentEnv(config)
    flat_env = FlatActionSpaceWrapper(env, config)
    env_model = MbagEnvModel(
        flat_env,
        config,
        player_index=player_index,
        include_action_mask_in_obs=include_action_mask_in_obs,
    )
    return env_model


register_env("MBAGAlphaZeroModel-v1", create_mbag_env_model)
