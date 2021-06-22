from typing import List, Tuple, Type

from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.mbag_env import MbagEnv, MbagConfigDict


MbagAgentConfig = Tuple[Type[MbagAgent], dict]


class MbagEvaluator(object):
    """
    Used to evaluate a (set of) MBAG agent(s).
    """

    def __init__(
        self, env_config: MbagConfigDict, agent_configs: List[MbagAgentConfig]
    ):
        self.env = MbagEnv(env_config)
        self.agents = [
            AgentClass(agent_config, env_config)
            for AgentClass, agent_config in agent_configs
        ]

    def rollout(self) -> float:
        """
        Run a single episode, returning the cumulative reward.
        """

        for agent in self.agents:
            agent.reset()
        all_obs = self.env.reset()
        done = False
        cumulative_reward = 0.0

        while not done:
            all_actions = [
                agent.get_action(obs) for agent, obs in zip(self.agents, all_obs)
            ]
            print(all_actions)
            all_obs, all_rewards, all_done, all_infos = self.env.step(all_actions)
            done = all_done[0]
            cumulative_reward += all_rewards[0]

        return cumulative_reward
