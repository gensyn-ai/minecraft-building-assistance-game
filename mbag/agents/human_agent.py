from abc import ABC
from typing import Any, List, cast
from mbag.environment.blocks import MinecraftBlocks
import numpy as np

from ..environment.types import (
    MbagAction,
    MbagActionType,
    MbagInfoDict,
    MbagObs,
    MbagActionTuple,
)
from ..environment.mbag_env import MbagConfigDict
from .mbag_agent import MbagAgent


class HumanAgent(MbagAgent):
    """
    An MBAG agent which chooses actions based on a queue that is fed in.
    """

    actions_queue: List[MbagAction]

    def reset(self) -> None:
        """
        This method is called whenever a new episode starts; it can be used to clear
        internal state or otherwise prepare for a new episode.
        """
        self.actions_queue = []

    def get_action_with_info(self, obs: MbagObs, info: MbagInfoDict) -> MbagActionTuple:
        """
        This should return an action ID to take in the environment. Either this or the
        get_action_*_distribution methods should be overridden.
        """
        self.actions_queue.extend(info.get("human_actions", []))

        action_type, block_location, block_id = MbagAction.NOOP, 0, 0
        if len(self.actions_queue > 0):
            action_type, block_location, block_id = self.actions_queue.pop(0)
        return action_type, block_location, block_id

    def get_state(self) -> List[np.ndarray]:
        """
        Get the current state of this agent as a list of zero or more numpy arrays.
        The agent should be able to be set back to its previous state by calling
        set_state with the return value of this method.
        """

        return [self.actions_queue]

    def set_state(self, state: List[np.ndarray]) -> None:
        """
        Restore the agent's state to what it was when get_state was called.
        """

        self.actions_queue = state[0]
