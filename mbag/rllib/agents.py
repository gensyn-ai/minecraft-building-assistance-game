import time
from typing import Iterable, List, Optional, cast

import numpy as np
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import TensorType
from typing_extensions import TypedDict

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.state import MbagStateDict
from mbag.environment.types import MbagInfoDict, MbagObs


class RllibMbagAgentConfigDict(TypedDict):
    policy: Policy
    min_action_interval: float
    """
    The minimum amount of time between actions, in seconds. If the agent is asked to
    take an action less than this amount of time after the last action, it will just
    return a NOOP.
    """


class RllibMbagAgent(MbagAgent):
    agent_config: RllibMbagAgentConfigDict
    state: List[TensorType]
    last_action_time: Optional[float]

    def __init__(self, agent_config: MbagConfigDict, env_config: MbagConfigDict):
        super().__init__(agent_config, env_config)

        self.policy = self.agent_config["policy"]
        self.min_action_interval = self.agent_config["min_action_interval"]
        self.action_mapping = MbagActionDistribution.get_action_mapping(self.env_config)

    def reset(self) -> None:
        self.state = self.policy.get_initial_state()
        self.last_action_time = None

    def get_action(self, obs: MbagObs, *, compute_actions_kwargs={}) -> MbagActionTuple:
        if self.last_action_time is not None:
            time_since_last_action = time.time() - self.last_action_time
            if time_since_last_action < self.min_action_interval:
                return (MbagAction.NOOP, 0, 0)

        obs_batch = tuple(obs_piece[None] for obs_piece in obs)
        state_batch = [state_piece[None] for state_piece in self.state]
        state_out_batch: List[TensorType]
        action_batch: Iterable[np.ndarray]
        action_batch, state_out_batch, info = self.policy.compute_actions(
            obs_batch, state_batch, explore=False, **compute_actions_kwargs
        )
        self.state = [state_piece[0] for state_piece in state_out_batch]

        self.last_action_time = time.time()

        if isinstance(action_batch, tuple):
            return cast(
                MbagActionTuple,
                tuple(int(action_piece[0]) for action_piece in action_batch),
            )
        else:
            # Flat actions.
            return cast(MbagActionTuple, tuple(self.action_mapping[action_batch[0]]))

    def get_state(self) -> List[np.ndarray]:
        return [np.array(state_part) for state_part in self.state]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.state = [state_part for state_part in state]


class RllibAlphaZeroAgentConfigDict(RllibMbagAgentConfigDict):
    player_index: str


class FakeEpisode(object):
    def __init__(self, *, user_data):
        self.user_data = user_data
        self.length = 0


class RllibAlphaZeroAgent(RllibMbagAgent):
    agent_config: RllibAlphaZeroAgentConfigDict

    def __init__(self, agent_config: MbagConfigDict, env_config: MbagConfigDict):
        super().__init__(agent_config, env_config)

        self.policy.config["player_index"] = self.agent_config["player_index"]

    def get_action_with_info_and_env_state(
        self, obs: MbagObs, info: Optional[MbagInfoDict], env_state: MbagStateDict
    ) -> MbagActionTuple:
        episode = FakeEpisode(user_data={"state": env_state})
        return super().get_action(obs, compute_actions_kwargs={"episodes": [episode]})
