from dataclasses import dataclass
from typing import List, Tuple, Type

from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.mbag_env import MbagEnv, MbagConfigDict
from mbag.environment.types import MbagObs


MbagAgentConfig = Tuple[Type[MbagAgent], dict]


@dataclass
class EpisodeInfo:
    cumulative_reward: float
    length: int
    last_obs: List[MbagObs]


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

    def rollout(self) -> EpisodeInfo:
        """
        Run a single episode, returning the cumulative reward.
        """

        for agent in self.agents:
            agent.reset()
        all_obs = self.env.reset()
        done = False
        cumulative_reward = 0.0
        timestep = 0
        if self.force_get_set_state:
            agent_states = [agent.get_state() for agent in self.agents]

        while not done:
            if self.force_get_set_state:
                for agent, state in zip(self.agents, agent_states):
                    agent.reset()
                    agent.set_state(state)
            all_actions = [
                agent.get_action(obs) for agent, obs in zip(self.agents, all_obs)
            ]
            all_obs, all_rewards, all_done, all_infos = self.env.step(all_actions)
            done = all_done[0]
            cumulative_reward += all_rewards[0]
            timestep += 1
            if self.force_get_set_state:
                agent_states = [agent.get_state() for agent in self.agents]

        return EpisodeInfo(
            cumulative_reward=cumulative_reward,
            length=timestep,
            last_obs=all_obs,
        )
