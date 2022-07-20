"""
RLLib-compatible MBAG environment.
"""

from typing import cast, Tuple
import numpy as np
from gym import spaces

from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.mbag_env import MbagConfigDict, MbagEnv


class MbagRllibEnv(MultiAgentEnv):
    action_space: spaces.Space

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = list(self._agent_ids)
        return {agent_id: self.action_space.sample() for agent_id in agent_ids}


class MbagMultiAgentEnv(MbagRllibEnv):
    def __init__(self, config):
        super().__init__()

        self.env = MbagEnv(cast(MbagConfigDict, config))

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._agent_ids = {
            self._agent_id(player_index)
            for player_index in range(config["num_players"])
        }

    def _agent_id(self, player_index: int) -> str:
        return f"player_{player_index}"

    def _dict_to_list(self, multi_agent_dict: MultiAgentDict) -> list:
        return [
            multi_agent_dict[self._agent_id(player_index)]
            for player_index in range(self.env.config["num_players"])
        ]

    def _list_to_dict(self, multi_agent_list: list) -> MultiAgentDict:
        return {
            self._agent_id(player_index): element
            for player_index, element in enumerate(multi_agent_list)
        }

    def reset(self) -> MultiAgentDict:
        obs_list = self.env.reset()
        return self._list_to_dict(obs_list)

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        action_list = self._dict_to_list(action_dict)
        obs_list, reward_list, done_list, info_list = self.env.step(action_list)

        obs_dict = self._list_to_dict(obs_list)
        reward_dict = self._list_to_dict(reward_list)
        done_dict = self._list_to_dict(done_list)
        done_dict["__all__"] = all(done_list)
        info_dict = self._list_to_dict(info_list)

        return obs_dict, reward_dict, done_dict, info_dict

    def render(self):
        return None


register_env("MBAG-v1", lambda config: MbagMultiAgentEnv(config))


class FlatActionSpaceWrapper(MbagRllibEnv):
    env: MultiAgentEnv
    action_space: spaces.Space

    def __init__(
        self,
        env: MultiAgentEnv,
        config: MbagConfigDict,
        include_action_mask_in_obs=False,
    ):
        super().__init__()

        self.env = env
        self.config = config
        self._agent_ids = self.env._agent_ids
        self.include_action_mask_in_obs = include_action_mask_in_obs
        self.action_mapping = MbagActionDistribution.get_action_mapping(self.config)

        num_flat_actions, _ = self.action_mapping.shape
        self.action_space = spaces.Discrete(num_flat_actions)
        if include_action_mask_in_obs:
            self.observation_space = spaces.Dict(
                {
                    "obs": self.env.observation_space,
                    "action_mask": spaces.Box(
                        np.zeros(num_flat_actions, dtype=bool),
                        np.ones(num_flat_actions, dtype=bool),
                    ),
                }
            )
        else:
            self.observation_space = self.env.observation_space

    def reset(self) -> MultiAgentDict:
        return cast(MultiAgentDict, self.env.reset())

    def step(
        self, flat_action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        action_dict = {
            agent_id: tuple(self.action_mapping[action])
            for agent_id, action in flat_action_dict.items()
        }
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        if self.include_action_mask_in_obs:
            for agent_id in obs_dict:
                obs = obs_dict[agent_id]
                obs_dict[agent_id] = {
                    "obs": obs,
                    "action_mask": MbagActionDistribution.get_mask_flat(
                        self.config, obs
                    ),
                }
        return obs_dict, reward_dict, done_dict, info_dict


register_env(
    "MBAGFlatActions-v1",
    lambda config: FlatActionSpaceWrapper(MbagMultiAgentEnv(config), config),
)
register_env(
    "MBAGFlatActionsAlphaZero-v1",
    lambda config: FlatActionSpaceWrapper(
        MbagMultiAgentEnv(config), config, include_action_mask_in_obs=True
    ),
)


def unwrap_mbag_env(env: MultiAgentEnv) -> MbagEnv:
    while not isinstance(env, MbagEnv):
        env = env.env
    return env
