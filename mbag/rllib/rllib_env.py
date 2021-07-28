"""
RLLib-compatible MBAG environment.
"""

from typing import cast, Tuple
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mbag.environment.mbag_env import MbagConfigDict, MbagEnv


class MbagMultiAgentEnv(MultiAgentEnv):
    def __init__(self, **config):
        self.wrapped_env = MbagEnv(cast(MbagConfigDict, config))

        self.action_space = self.wrapped_env.action_space
        self.observation_space = self.wrapped_env.observation_space

    def _agent_id(self, player_index: int) -> str:
        return f"player_{player_index}"

    def _dict_to_list(self, multi_agent_dict: MultiAgentDict) -> list:
        return [
            multi_agent_dict[self._agent_id(player_index)]
            for player_index in range(self.wrapped_env.config["num_players"])
        ]

    def _list_to_dict(self, multi_agent_list: list) -> MultiAgentDict:
        return {
            self._agent_id(player_index): element
            for player_index, element in enumerate(multi_agent_list)
        }

    def reset(self) -> MultiAgentDict:
        obs_list = self.wrapped_env.reset()
        return self._list_to_dict(obs_list)

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        action_list = self._dict_to_list(action_dict)
        obs_list, reward_list, done_list, info_list = self.wrapped_env.step(action_list)

        obs_dict = self._list_to_dict(obs_list)
        reward_dict = self._list_to_dict(reward_list)
        done_dict = self._list_to_dict(done_list)
        done_dict["__all__"] = all(done_list)
        info_dict = self._list_to_dict(info_list)

        return obs_dict, reward_dict, done_dict, info_dict

    def render(self):
        return None


register_env("MBAG-v1", lambda config: MbagMultiAgentEnv(**config))
