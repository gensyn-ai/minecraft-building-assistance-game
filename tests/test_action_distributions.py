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
    env = MbagEnv(
        {"abilities": {"inf_blocks": False, "flying": True, "teleportation": True}}
    )
    world_obs, inventory_obs = env.reset()[0]
    inventory_obs[MinecraftBlocks.NAME2ID["planks"]] = 1
    obs_batch = np.array([world_obs, world_obs]), np.array(
        [inventory_obs, inventory_obs]
    )

    mask = MbagActionDistribution.get_block_id_mask(
        env.config,
        obs_batch,
        np.array([MbagAction.PLACE_BLOCK, MbagAction.GIVE_BLOCK]),
        np.array(
            [
                np.ravel_multi_index(
                    (1, 2, 1),
                    env.config["world_size"],
                ),
                np.ravel_multi_index(
                    (1, 2, 1),
                    env.config["world_size"],
                ),
            ]
        ),
    )
    assert mask[0, MinecraftBlocks.AIR] == False
    assert mask[0, MinecraftBlocks.BEDROCK] == False
    assert mask[0, MinecraftBlocks.NAME2ID["dirt"]] == False
    assert mask[0, MinecraftBlocks.NAME2ID["planks"]] == True
    assert mask[1, MinecraftBlocks.AIR] == False
    assert mask[1, MinecraftBlocks.BEDROCK] == False
    assert mask[1, MinecraftBlocks.NAME2ID["dirt"]] == False
    assert mask[1, MinecraftBlocks.NAME2ID["planks"]] == True


def test_block_id_unique():
    env = MbagEnv(
        {"abilities": {"inf_blocks": False, "flying": True, "teleportation": True}}
    )
    world_obs, inventory_obs = env.reset()[0]
    inventory_obs[MinecraftBlocks.NAME2ID["planks"]] = 1
    inventory_obs[MinecraftBlocks.NAME2ID["cobblestone"]] = 1

    obs_batch = np.array([world_obs, world_obs]), np.array(
        [inventory_obs, inventory_obs]
    )

    unique = MbagActionDistribution.get_block_id_unique(
        env.config,
        obs_batch,
        np.array([MbagAction.PLACE_BLOCK, MbagAction.GIVE_BLOCK]),
        np.array(
            [
                np.ravel_multi_index(
                    (1, 2, 1),
                    env.config["world_size"],
                ),
                np.ravel_multi_index(
                    (1, 2, 1),
                    env.config["world_size"],
                ),
            ]
        ),
    )
    assert unique[0, MinecraftBlocks.AIR] == unique[0, MinecraftBlocks.AIR]
    assert unique[0, MinecraftBlocks.NAME2ID["dirt"]] == unique[0, MinecraftBlocks.AIR]
    assert (
        unique[0, MinecraftBlocks.NAME2ID["planks"]]
        != unique[0, MinecraftBlocks.NAME2ID["cobblestone"]]
    )
    assert (
        unique[0, MinecraftBlocks.NAME2ID["planks"]] != unique[0, MinecraftBlocks.AIR]
    )
