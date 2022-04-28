import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.types import MbagAction


def test_action_type_location_mask():
    env = MbagEnv({})
    world_obs, inventory_obs = env.reset()[0]
    obs_batch = world_obs[None], inventory_obs[None]

    mask = MbagActionDistribution.get_action_type_location_mask(env.config, obs_batch)

    assert mask[0, MbagAction.PLACE_BLOCK, 1, 1, 1] == False
    assert mask[0, MbagAction.PLACE_BLOCK, 1, 2, 1] == True
    assert mask[0, MbagAction.PLACE_BLOCK, 1, 3, 1] == False

    assert mask[0, MbagAction.BREAK_BLOCK, 1, 0, 1] == False
    assert mask[0, MbagAction.BREAK_BLOCK, 1, 1, 1] == True
    assert mask[0, MbagAction.BREAK_BLOCK, 1, 2, 1] == False


def test_block_id_mask():
    env = MbagEnv({})
    world_obs, inventory_obs = env.reset()[0]
    obs_batch = world_obs[None], inventory_obs[None]

    mask = MbagActionDistribution.get_block_id_mask(
        env.config,
        obs_batch,
        np.array([MbagAction.PLACE_BLOCK]),
        np.array(
            [
                np.ravel_multi_index(
                    (1, 2, 1),
                    env.config["world_size"],
                )
            ]
        ),
    )
    assert mask[0, MinecraftBlocks.AIR] == False
    assert mask[0, MinecraftBlocks.BEDROCK] == False
    assert mask[0, MinecraftBlocks.NAME2ID["dirt"]] == True
