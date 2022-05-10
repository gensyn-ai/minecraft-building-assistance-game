import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.types import MbagAction, PLAYER_LOCATIONS


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

    # Moving is disabled by default.
    for move_action in MbagAction.MOVE_ACTION_TYPES:
        assert np.all(mask[0, move_action] == False)


def test_action_type_location_mask_no_teleportation():
    env = MbagEnv(
        {
            "num_players": 2,
            "abilities": {
                "teleportation": False,
                "flying": True,
                "inf_blocks": False,
            },
        }
    )
    world_obs, inventory_obs = env.reset()[0]
    obs_batch = world_obs[None], inventory_obs[None]
    mask = MbagActionDistribution.get_action_type_location_mask(env.config, obs_batch)

    # Player 1 should be at (0, 2, 0) and player 2 at (1, 2, 0).
    assert np.all(world_obs[PLAYER_LOCATIONS][0, 2:4, 0] == 1)
    assert np.all(world_obs[PLAYER_LOCATIONS][1, 2:4, 0] == 2)

    # They shouldn't be able to place/break blocks more than 3 blocks away.
    assert np.all(mask[0, MbagAction.PLACE_BLOCK, 4, :, :] == False)
    assert np.all(mask[0, MbagAction.PLACE_BLOCK, :, :, 4] == False)
    assert np.all(mask[0, MbagAction.BREAK_BLOCK, 4, :, :] == False)
    assert np.all(mask[0, MbagAction.BREAK_BLOCK, :, :, 4] == False)

    # Should only be able to give to a location where a player is.
    assert mask[0, MbagAction.GIVE_BLOCK, 1, 2, 0] == True
    assert mask[0, MbagAction.GIVE_BLOCK, 2, 2, 0] == False

    world_obs, inventory_obs = env.step(
        [(MbagAction.NOOP, 0, 0), (MbagAction.MOVE_POS_X, 0, 0)]
    )[0][0]
    obs_batch = world_obs[None], inventory_obs[None]
    mask = MbagActionDistribution.get_action_type_location_mask(env.config, obs_batch)

    # Player 2 should now be at (2, 2, 0).
    assert np.all(world_obs[PLAYER_LOCATIONS][2, 2:4, 0] == 2)
    # Now it should be impossible to give blocks since player 2 is too far away.
    assert np.all(mask[0, MbagAction.GIVE_BLOCK] == False)

    assert False


def test_action_type_location_unique():
    env = MbagEnv(
        {
            "num_players": 2,
            "abilities": {
                "teleportation": False,
                "flying": True,
                "inf_blocks": False,
            },
        }
    )
    world_obs, inventory_obs = env.reset()[0]
    obs_batch = world_obs[None], inventory_obs[None]

    unique = MbagActionDistribution.get_action_type_location_unique(
        env.config, obs_batch
    )

    # Placing blocks at different locations is different.
    assert (
        unique[0, MbagAction.PLACE_BLOCK, 1, 2, 1]
        != unique[0, MbagAction.PLACE_BLOCK, 2, 2, 1]
    )

    # Moving should be the same across locations.
    for move_action in MbagAction.MOVE_ACTION_TYPES:
        action_unique = unique[0, move_action]
        action_unique = action_unique[action_unique != 0]
        assert len(np.unique(action_unique)[0]) == 1
        # ...and it should not overlap with other action types.
        assert action_unique[0] not in unique[0, :move_action]

    # more tests needed...


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
