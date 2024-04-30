from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import TensorType

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.actions import MbagActionTuple
from mbag.environment.types import MbagObs

# Backwards compatibility for importing PPO classes which used to be in this module.
from .ppo import MbagPPO, MbagPPOConfig, MbagPPOTorchPolicy  # noqa: F401


class MbagAgentPolicy(Policy):
    """
    An RLlib policy that selects actions based on an MBAG agent instance.
    """

    agent: MbagAgent
    config: Dict[str, Any]

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config,
    ):
        super().__init__(observation_space, action_space, config)
        self.agent = config["mbag_agent"]
        self.force_seed: Optional[int] = config.get("force_seed", None)
        self.exploration = self._create_exploration()
        self.flat_actions = isinstance(self.action_space, spaces.Discrete)

        self.view_requirements[SampleBatch.ACTION_DIST_INPUTS] = ViewRequirement()

    def get_initial_state(self):
        self.agent.reset(seed=self.force_seed)
        return self.agent.get_state()

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        for batch_key, view_requirement in self.view_requirements.items():
            if batch_key.startswith("state_in_"):
                view_requirement.batch_repeat_value = 1

        unflattened_obs_batch = restore_original_dimensions(
            obs_batch,
            obs_space=self.observation_space,
            tensorlib=np,
        )

        actions: List[MbagActionTuple] = []
        action_dist_inputs: List[np.ndarray] = []
        new_states: List[List[np.ndarray]] = []

        assert state_batches is not None

        obs: MbagObs
        prev_state: Sequence[np.ndarray]
        for obs, prev_state in zip(
            zip(*unflattened_obs_batch),
            (
                zip(*state_batches)
                if state_batches != []
                else [[] for _ in range(len(obs_batch))]
            ),
        ):
            self.agent.set_state(list(prev_state))
            try:
                action, action_dist = self.agent.get_action_with_distribution(obs)
                if self.flat_actions:
                    action_dist = MbagActionDistribution.to_flat(
                        self.config["env_config"], action_dist[None]
                    )[0]
                action_dist_inputs.append(np.log(action_dist))
            except NotImplementedError:
                action = self.agent.get_action(obs)
            actions.append(action)
            new_states.append(self.agent.get_state())

        action_array: Union[np.ndarray, Tuple[np.ndarray, ...]]
        if self.flat_actions:
            action_array = np.array(
                [
                    MbagActionDistribution.get_flat_action(
                        self.config["env_config"], action
                    )
                    for action in actions
                ]
            )
        else:
            action_array = tuple(
                np.array([action[action_part] for action in actions])
                for action_part in range(3)
            )
        state_arrays = [
            np.array([new_state[state_part] for new_state in new_states])
            for state_part in range(len(state_batches))
        ]
        extra_fetches = {}
        if action_dist_inputs:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = np.stack(
                action_dist_inputs, axis=0
            )
        return action_array, state_arrays, extra_fetches  # type: ignore

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
