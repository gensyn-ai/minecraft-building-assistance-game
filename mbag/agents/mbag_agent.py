from abc import ABC
import numpy as np

from ..environment.types import MbagAction, MbagObs, MbagActionId
from ..environment.mbag_env import MbagConfigDict


class MbagAgent(ABC):
    """
    An MBAG agent which chooses actions based on observations.
    """

    agent_config: dict
    env_config: MbagConfigDict

    def __init__(self, agent_config: dict, env_config: MbagConfigDict):
        self.agent_config = agent_config
        self.env_config = env_config

    def reset(self) -> None:
        """
        This method is called whenever a new episode starts; it can be used to clear
        internal state or otherwise prepare for a new episode.
        """

        pass

    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        """
        This should return a distribution over actions that sums to 1. Either this or
        get_action() should be overridden.
        """

        action_shape = MbagAction.get_action_shape(self.env_config["world_size"])
        uniform_dist = np.ones(action_shape)
        uniform_dist /= uniform_dist.sum()
        return uniform_dist

    def get_action(self, obs: MbagObs) -> MbagActionId:
        """
        This should return an action ID to take in the environment. Either this or
        get_action() should be overridden. See MbagAction for more information.
        """

        action_dist = self.get_action_distribution(obs)
        return int(np.random.multinomial(1, action_dist.flat).argmax())
