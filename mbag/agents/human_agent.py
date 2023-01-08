import logging
from typing import List, Optional

import numpy as np

from ..environment.types import MbagAction, MbagActionTuple, MbagInfoDict, MbagObs
from .mbag_agent import MbagAgent

logger = logging.getLogger(__name__)


class HumanAgent(MbagAgent):
    """
    An MBAG agent which chooses actions based on a queue that is fed in.
    """

    actions_queue: List[MbagActionTuple]
    last_action: Optional[MbagAction]

    def reset(self) -> None:
        """
        This method is called whenever a new episode starts; it can be used to clear
        internal state or otherwise prepare for a new episode.
        """
        self.actions_queue = []
        self.last_action = None

    def get_action_with_info(self, obs: MbagObs, info: MbagInfoDict) -> MbagActionTuple:
        """
        This should return an action ID to take in the environment. Either this or the
        get_action_*_distribution methods should be overridden.
        """

        if self.last_action is not None:
            if info["action"].to_tuple() != self.last_action.to_tuple():
                logger.error(
                    f"human action did not succeed: expected action "
                    f"{self.last_action} but env reported {info['action']}"
                )

        self.actions_queue.extend(info.get("human_actions", []))

        action_tuple: MbagActionTuple = MbagAction.NOOP, 0, 0
        if len(self.actions_queue) > 0:
            action_tuple = self.actions_queue.pop(0)
            action = MbagAction(action_tuple, self.env_config["world_size"])

            logger.info(f"human action being replayed: {action}")

        self.last_action = MbagAction(action_tuple, self.env_config["world_size"])

        return action_tuple

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
