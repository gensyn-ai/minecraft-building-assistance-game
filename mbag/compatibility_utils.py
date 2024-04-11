import copy
import pickle
from typing import Union

from mbag.environment.config import MbagConfigDict, RewardsConfigDict


class OldHumanDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "mbag.environment.types" and name in [
            "MbagActionTuple",
            "MbagAction",
            "MbagActionType",
        ]:
            module = "mbag.environment.actions"
        return super().find_class(module, name)


def convert_old_rewards_config_to_new(
    rewards_config: Union[RewardsConfigDict, dict]
) -> RewardsConfigDict:
    own_reward_prop_start = rewards_config.get("own_reward_prop", 0.0)
    own_reward_prop_horizon = rewards_config.get("own_reward_prop_horizon", None)
    if own_reward_prop_horizon is None:
        own_reward_prop = own_reward_prop_start
    else:
        if (
            not isinstance(own_reward_prop_horizon, (int, float))
            or int(own_reward_prop_horizon) != own_reward_prop_horizon
        ):
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


def convert_old_config_to_new(
    old_config: MbagConfigDict,
) -> MbagConfigDict:
    mbag_config = copy.deepcopy(old_config)
    if "rewards" in mbag_config:
        mbag_config["rewards"] = convert_old_rewards_config_to_new(
            mbag_config["rewards"]
        )
    for player_config in mbag_config.get("players", []):
        if "rewards" in player_config:
            player_config["rewards"] = convert_old_rewards_config_to_new(
                player_config["rewards"]
            )
    return mbag_config
