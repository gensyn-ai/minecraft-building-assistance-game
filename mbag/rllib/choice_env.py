from typing import cast, Tuple
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mbag.environment.mbag_env import MbagConfigDict, MbagEnv
from .rllib_env import MbagMultiAgentEnv
from mbag.agents.action_distributions import MbagActionDistribution

import numpy as np


class ChoiceRewardWrapper(MbagMultiAgentEnv):
    def __init__(self, **config):
        # self.choice_reward_weight = config.pop("choice_reward_weight")
        self.mbag_env = MbagMultiAgentEnv(**config)

    def reset(self):
        return self.mbag_env.reset()

    def step(self, action_dict):
        obs_dict, reward_dict, done_dict, info_dict = self.mbag_env.step(action_dict)
        world_obs, inventory_obs = obs_dict[self.mbag_env._agent_id(0)]
        reward_dict[self.mbag_env._agent_id(1)] = self.calculate_choices(
            world_obs, inventory_obs, action_dict[self.mbag_env._agent_id(0)]
        )

        return obs_dict, reward_dict, done_dict, info_dict

    def calculate_choices(self, world_obs, inventory_obs, action) -> int:
        obs_batch = world_obs[None], inventory_obs[None]
        location_choices = MbagActionDistribution.get_action_type_location_unique(
            self.mbag_env.wrapped_env.config, obs_batch
        )

        item_choices = MbagActionDistribution.get_block_id_unique(
            self.mbag_env.wrapped_env.config,
            obs_batch,
            np.array([action.action_type]),
            np.array([action.block_location]),
        )

        return (
            np.count_unique(location_choices).size + np.count_unique(item_choices).size
        )


register_env(
    "MBAG-ChoiceRewardWrapper-v1", lambda config: ChoiceRewardWrapper(**config)
)
