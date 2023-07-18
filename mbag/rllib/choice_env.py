from ray.tune.registry import register_env

from mbag.agents.action_distributions import MbagActionDistribution

from .rllib_env import MbagMultiAgentEnv


class ChoiceRewardWrapper(MbagMultiAgentEnv):
    def __init__(self, config):
        # self.choice_reward_weight = config.pop("choice_reward_weight")
        self.mbag_env = MbagMultiAgentEnv(config)

    def reset(self):
        return self.mbag_env.reset()

    def step(self, action_dict):
        (
            obs_dict,
            reward_dict,
            terminated_dict,
            truncated_dict,
            info_dict,
        ) = self.mbag_env.step(action_dict)
        (
            world_obs,
            inventory_obs,
            timestep,
        ) = obs_dict[self.mbag_env._agent_id(0)]
        reward_dict[self.mbag_env._agent_id(1)] = self.calculate_choices(
            world_obs, inventory_obs, timestep, action_dict[self.mbag_env._agent_id(0)]
        )

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def calculate_choices(self, world_obs, inventory_obs, timestep, action) -> int:
        obs_batch = world_obs[None], inventory_obs[None], timestep[None]
        flat_mask = MbagActionDistribution.get_mask_flat(
            self.mbag_env.env.config, obs_batch
        )
        return int(flat_mask.astype(int).sum())


register_env(
    "MBAG-ChoiceRewardWrapper-v1", lambda config: ChoiceRewardWrapper(**config)
)
