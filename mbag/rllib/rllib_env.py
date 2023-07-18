"""
RLLib-compatible MBAG environment.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    from .planning import MbagEnvModel

from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict
from ray.tune.registry import register_env

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.mbag_env import MbagConfigDict, MbagEnv, MbagStateDict


class MbagRllibWrapper(MultiAgentEnv):
    action_space: spaces.Space
    env: Union["MbagRllibWrapper", MbagEnv]

    def action_space_sample(
        self, agent_ids: Optional[List[AgentID]] = None
    ) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = list(self._agent_ids)
        return {agent_id: self.action_space.sample() for agent_id in agent_ids}

    def get_state(self) -> MbagStateDict:
        return self.env.get_state()

    def set_state_no_obs(self, state: MbagStateDict) -> None:
        self.env.set_state_no_obs(state)

    def set_state(self, state: MbagStateDict) -> MultiAgentDict:
        return cast(MbagRllibWrapper, self.env).set_state(state)


class MbagMultiAgentEnv(MbagRllibWrapper):
    env: MbagEnv

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

    def reset(self, **kwargs):
        obs_list = self.env.reset()
        obs_dict = self._list_to_dict(obs_list)
        info_dict: MultiAgentDict = {agent_id: {} for agent_id in obs_dict}
        return obs_dict, info_dict

    def step(  # type: ignore
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        action_list = self._dict_to_list(action_dict)
        obs_list, reward_list, done_list, info_list = self.env.step(action_list)

        obs_dict = self._list_to_dict(obs_list)
        reward_dict = self._list_to_dict(reward_list)
        done_dict = self._list_to_dict(done_list)
        done_dict["__all__"] = all(done_list)
        info_dict = self._list_to_dict(info_list)

        terminated_dict = done_dict
        truncated_dict = {agent_id: False for agent_id in done_dict}

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def render(self):
        return None

    def set_state(self, state: MbagStateDict) -> MultiAgentDict:
        obs_list = self.env.set_state(state)
        return self._list_to_dict(obs_list)


register_env("MBAG-v1", lambda config: MbagMultiAgentEnv(config))


class FlatActionSpaceWrapper(MbagRllibWrapper):
    env: MbagRllibWrapper
    action_space: spaces.Space

    def __init__(
        self,
        env,
        config: MbagConfigDict,
    ):
        super().__init__()

        self.env = env
        self.config = config
        self._agent_ids = self.env._agent_ids
        self.action_mapping = MbagActionDistribution.get_action_mapping(self.config)

        num_flat_actions, _ = self.action_mapping.shape
        self.action_space = spaces.Discrete(num_flat_actions)
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, flat_action_dict: MultiAgentDict):
        action_dict = {
            agent_id: tuple(self.action_mapping[action])
            for agent_id, action in flat_action_dict.items()
        }
        (
            obs_dict,
            reward_dict,
            terminated_dict,
            truncated_dict,
            info_dict,
        ) = self.env.step(action_dict)
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict


register_env(
    "MBAGFlatActions-v1",
    lambda config: FlatActionSpaceWrapper(MbagMultiAgentEnv(config), config),
)


def unwrap_mbag_env(env: Union["MbagEnvModel", MbagRllibWrapper, MbagEnv]) -> MbagEnv:
    while not isinstance(env, MbagEnv):
        env = env.env
    return env
