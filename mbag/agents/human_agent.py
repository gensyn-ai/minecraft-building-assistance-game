import logging
from typing import Optional

from ..environment.actions import MbagAction, MbagActionTuple
from ..environment.types import MbagInfoDict, MbagObs
from .mbag_agent import MbagAgent

logger = logging.getLogger(__name__)


class HumanAgent(MbagAgent):
    """
    An MBAG agent which chooses actions based on a queue that is fed in.
    """

    last_action: Optional[MbagAction]

    def reset(self, **kwargs) -> None:
        """
        This method is called whenever a new episode starts; it can be used to clear
        internal state or otherwise prepare for a new episode.
        """

        super().reset(**kwargs)

        self.last_action = None

    def get_action_with_info(
        self, obs: MbagObs, info: Optional[MbagInfoDict]
    ) -> MbagActionTuple:
        """
        This should return an action ID to take in the environment. Either this or the
        get_action_*_distribution methods should be overridden.
        """

        if info is None:
            assert self.last_action is None
            action_tuple = (MbagAction.NOOP, 0, 0)
        else:
            if self.last_action is not None:
                if info["action"].to_tuple() != self.last_action.to_tuple():
                    logger.error(
                        f"human action did not succeed: expected action "
                        f"{self.last_action} but env reported {info['action']}"
                    )
            action_tuple = info["human_action"]

        action = MbagAction(action_tuple, self.env_config["world_size"])
        if action.action_type != MbagAction.NOOP:
            logger.info(f"human action being replayed: {action}")
        self.last_action = action

        return action_tuple
