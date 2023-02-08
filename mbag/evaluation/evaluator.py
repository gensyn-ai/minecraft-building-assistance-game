import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type

from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.mbag_env import MbagConfigDict, MbagEnv
from mbag.environment.types import MbagActionTuple, MbagInfoDict, MbagObs

logger = logging.getLogger(__name__)

MbagAgentConfig = Tuple[Type[MbagAgent], Any]
"""
An MbagAgent subclass together with the agent config for that agent.
"""


@dataclass
class EpisodeInfo:
    reward_history: List[float]
    cumulative_reward: float
    length: int
    obs_history: List[List[MbagObs]]
    last_obs: List[MbagObs]
    info_history: List[List[MbagInfoDict]]
    last_infos: List[MbagInfoDict]

    def to_json(self) -> dict:
        return {
            "cumulative_reward": self.cumulative_reward,
            "length": self.length,
            "last_infos": self.last_infos,
        }

    def toJSON(self) -> dict:  # noqa: N802
        return {
            "cumulative_reward": self.cumulative_reward,
            "length": self.length,
            "info_history": self.info_history,
            # "obs_history": self.obs_history,
        }


class MbagEvaluator(object):
    """
    Used to evaluate a (set of) MBAG agent(s).
    """

    def __init__(
        self,
        env_config: MbagConfigDict,
        agent_configs: List[MbagAgentConfig],
        *,
        force_get_set_state=False,
        return_on_exception=False,
    ):
        self.env = MbagEnv(env_config)
        env_config = self.env.config
        self.agents = [
            agent_class(agent_config, env_config)
            for agent_class, agent_config in agent_configs
        ]
        self.force_get_set_state = force_get_set_state
        self.return_on_exception = return_on_exception

    def rollout(self) -> EpisodeInfo:
        """
        Run a single episode, returning the cumulative reward.
        """
        try:
            for agent in self.agents:
                agent.reset()
            all_obs = self.env.reset()
            previous_infos: Optional[List[MbagInfoDict]] = None
            done = False
            timestep = 0
            if self.force_get_set_state:
                agent_states = [agent.get_state() for agent in self.agents]

            # should the initial setting be included?
            reward_history = [0.0]
            obs_history = [all_obs]
            info_history: List[List[MbagInfoDict]] = []

            while not done:
                if self.force_get_set_state:
                    for agent, state in zip(self.agents, agent_states):
                        agent.reset()
                        agent.set_state(state)

                all_actions: List[MbagActionTuple] = []
                for agent_index, agent in enumerate(self.agents):
                    previous_info: Optional[MbagInfoDict] = None
                    if previous_infos is not None:
                        previous_info = previous_infos[agent_index]
                    obs = all_obs[agent_index]
                    action = agent.get_action_with_info(obs, previous_info)
                    all_actions.append(action)

                all_obs, all_rewards, all_done, all_infos = self.env.step(all_actions)
                done = all_done[0]
                reward_history.append(all_rewards[0])
                obs_history.append(all_obs)
                info_history.append(all_infos)
                timestep += 1

                if self.force_get_set_state:
                    agent_states = [agent.get_state() for agent in self.agents]
                self.previous_infos = all_infos

            episode_info = EpisodeInfo(
                reward_history=reward_history,
                cumulative_reward=sum(reward_history),
                length=timestep,
                last_obs=all_obs,
                last_infos=all_infos,
                obs_history=obs_history,
                info_history=info_history,
            )
            return episode_info
        except Exception as exception:
            if self.return_on_exception:
                logger.error(exception)
                return episode_info
            else:
                raise exception
