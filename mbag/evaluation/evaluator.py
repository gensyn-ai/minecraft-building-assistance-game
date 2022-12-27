from dataclasses import dataclass
from typing import Any, List, Tuple, Type

from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.mbag_env import MbagEnv, MbagConfigDict
from mbag.environment.types import MbagObs, MbagInfoDict


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
    ):
        self.env = MbagEnv(env_config)
        self.agents = [
            agent_class(agent_config, env_config)
            for agent_class, agent_config in agent_configs
        ]
        self.force_get_set_state = force_get_set_state
        self.previous_infos = [{} for _ in self.agents]
        self.episodes = []

    def rollout(self) -> EpisodeInfo:
        """
        Run a single episode, returning the cumulative reward.
        """

        for agent in self.agents:
            agent.reset()
        all_obs = self.env.reset()
        done = False
        timestep = 0
        if self.force_get_set_state:
            agent_states = [agent.get_state() for agent in self.agents]

        # should the initial setting be included?
        reward_history = [0.0]
        obs_history = [all_obs]
        info_history = [self.previous_infos]

        while not done:
            if self.force_get_set_state:
                for agent, state in zip(self.agents, agent_states):
                    agent.reset()
                    agent.set_state(state)
            all_actions = [
                agent.get_action_with_info(obs, info)
                for agent, obs, info in zip(self.agents, all_obs, self.previous_infos)
            ]
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
        self.episodes.append(episode_info)
        return episode_info

    def log_episodes(self):
        print(self.episodes)
