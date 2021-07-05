from typing import List, Tuple
import gym
import numpy as np
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.models.modelv2 import restore_original_dimensions

from mbag.environment.types import MbagActionTuple, MbagObs
from mbag.agents.mbag_agent import MbagAgent


class MbagAgentPolicy(Policy):
    """
    An RLlib policy that selects actions based on an MBAG agent instance.
    """

    agent: MbagAgent

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict,
    ):
        super().__init__(observation_space, action_space, config)
        self.agent = config["mbag_agent"]
        self.exploration = self._create_exploration()

    def get_initial_state(self) -> List[TensorType]:
        self.agent.reset()
        return self.agent.get_state()

    def compute_actions(
        self,
        obs_batch: List[np.ndarray],
        state_batches: List[np.ndarray],
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        self.view_requirements["state_in_0"].batch_repeat_value = 1

        unflattened_obs_batch = restore_original_dimensions(
            obs_batch,
            obs_space=self.observation_space,
            tensorlib=np,
        )

        actions: List[MbagActionTuple] = []
        new_states: List[List[np.ndarray]] = []

        obs: MbagObs
        prev_state: Tuple[np.ndarray, ...]
        for obs, prev_state in zip(zip(*unflattened_obs_batch), zip(*state_batches)):
            self.agent.set_state(list(prev_state))
            action = self.agent.get_action(obs)
            actions.append(action)
            new_states.append(self.agent.get_state())

        action_arrays = tuple(
            np.array([action[action_part] for action in actions])
            for action_part in range(3)
        )
        state_arrays = [
            np.array([new_state[state_part] for new_state in new_states], dtype=float)
            for state_part in range(len(state_batches))
        ]
        return action_arrays, state_arrays, {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
