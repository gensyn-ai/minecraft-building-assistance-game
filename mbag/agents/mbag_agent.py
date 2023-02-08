from abc import ABC
from typing import Any, List, Tuple, cast

import numpy as np

from mbag.environment.mbag_env import MbagConfigDict, MbagEnv
from mbag.environment.types import MbagActionTuple, MbagObs

from .action_distributions import MbagActionDistribution


class MbagAgent(ABC):
    """
    An MBAG agent which chooses actions based on observations.
    """

    agent_config: Any
    env_config: MbagConfigDict

    def __init__(self, agent_config: Any, env_config: MbagConfigDict):
        self.agent_config = agent_config
        self.env_config = MbagEnv.get_config(env_config)
        self.action_mapping = MbagActionDistribution.get_action_mapping(self.env_config)

    def reset(self) -> None:
        """
        This method is called whenever a new episode starts; it can be used to clear
        internal state or otherwise prepare for a new episode.
        """

        pass

    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        """
        This should return a distribution over actions of shape
        (NUM_CHANNELS, width, height, depth). See MbagActionDistribution for details.
        """

        raise NotImplementedError()

    def get_action_with_distribution(
        self, obs: MbagObs
    ) -> Tuple[MbagActionTuple, np.ndarray]:
        """
        Calculates the action distribution for this observation with
        get_action_distribution, then
        """

        action_distribution = self.get_action_distribution(obs)
        flat_action_distribution = MbagActionDistribution.to_flat(
            self.env_config, action_distribution[None]
        )[0]
        flat_action = np.argmax(np.random.multinomial(1, flat_action_distribution))
        action_tuple: MbagActionTuple = cast(
            MbagActionTuple, tuple(self.action_mapping[flat_action])
        )
        return action_tuple, action_distribution

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        """
        This should return an action ID to take in the environment. Either this or the
        get_action_distribution method should be overridden.
        """

        action, _ = self.get_action_with_distribution(obs)
        return action

    def get_state(self) -> List[np.ndarray]:
        """
        Get the current state of this agent as a list of zero or more numpy arrays.
        The agent should be able to be set back to its previous state by calling
        set_state with the return value of this method.
        """

        return []

    def set_state(self, state: List[np.ndarray]) -> None:
        """
        Restore the agent's state to what it was when get_state was called.
        """

        pass
