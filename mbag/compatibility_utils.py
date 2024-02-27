import pickle

from mbag.environment.config import RewardsConfigDict


class OldHumanDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "mbag.environment.types" and name in [
            "MbagActionTuple",
            "MbagAction",
            "MbagActionType",
        ]:
            module = "mbag.environment.actions"
        return super().find_class(module, name)


def convert_old_rewards_config_to_new(rewards_config: dict) -> RewardsConfigDict:
    own_reward_prop_start = rewards_config.get("own_reward_prop", 0.0)
    own_reward_prop_horizon = rewards_config.get("own_reward_prop_horizon", None)
    if own_reward_prop_horizon is None:
        own_reward_prop = own_reward_prop_start
    else:
        if int(own_reward_prop_horizon) != own_reward_prop_horizon:
            raise ValueError(
                f"own_reward_prop_horizon must be an integer, got {own_reward_prop_horizon}"
            )
        own_reward_prop = [(0, own_reward_prop_start), (own_reward_prop_horizon, 0.0)]

    return RewardsConfigDict(
        noop=rewards_config.get("noop", 0.0),
        action=rewards_config.get("action", 0.0),
        place_wrong=rewards_config.get("place_wrong", 0.0),
        own_reward_prop=own_reward_prop,
        get_resources=rewards_config.get("get_resources", 0.0),
    )
