import copy

import numpy as np
import pytest
import torch

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import DEFAULT_CONFIG, MbagEnv
from mbag.environment.types import CURRENT_BLOCKS, PLAYER_LOCATIONS, MbagAction


def test_mapping():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = (5, 5, 5)

    config["abilities"] = {
        "teleportation": True,
        "flying": True,
        "inf_blocks": True,
    }
    mapping = MbagActionDistribution.get_action_mapping(config)
    assert mapping.shape == (
        1  # NOOP
        + 125 * MinecraftBlocks.NUM_BLOCKS  # PLACE_BLOCK
        + 125,  # BREAK_BLOCK
        3,
    )
    assert mapping[0].tolist() == [MbagAction.NOOP, 0, 0]
    assert mapping[1].tolist() == [MbagAction.PLACE_BLOCK, 0, 0]
    assert mapping[2].tolist() == [MbagAction.PLACE_BLOCK, 1, 0]
    assert mapping[1 + 125].tolist() == [MbagAction.PLACE_BLOCK, 0, 1]
    assert mapping[1 + 125 * MinecraftBlocks.NUM_BLOCKS].tolist() == [
        MbagAction.BREAK_BLOCK,
        0,
        0,
    ]
    assert mapping[2 + 125 * MinecraftBlocks.NUM_BLOCKS].tolist() == [
        MbagAction.BREAK_BLOCK,
        1,
        0,
    ]

    config["abilities"] = {
        "teleportation": False,
        "flying": True,
        "inf_blocks": False,
    }
    mapping_all_abilities = mapping
    mapping = MbagActionDistribution.get_action_mapping(config)
    assert mapping.shape == (
        1  # NOOP
        + 125 * MinecraftBlocks.NUM_BLOCKS  # PLACE_BLOCK
        + 125  # BREAK_BLOCK
        + 6  # movement actions
        + 125 * MinecraftBlocks.NUM_BLOCKS,  # GIVE_BLOCK
        3,
    )
    assert (
        mapping[: mapping_all_abilities.shape[0]].tolist()
        == mapping_all_abilities.tolist()
    )
    assert mapping[1 + 125 * (MinecraftBlocks.NUM_BLOCKS + 1)].tolist() == [
        MbagAction.MOVE_POS_X,
        0,
        0,
    ]
    assert mapping[6 + 125 * (MinecraftBlocks.NUM_BLOCKS + 1)].tolist() == [
        MbagAction.MOVE_NEG_Z,
        0,
        0,
    ]


def test_mask():
    env = MbagEnv(
        {"abilities": {"teleportation": True, "inf_blocks": True, "flying": True}}
    )
    world_obs, inventory_obs, timestep = env.reset()[0]
    planks = MinecraftBlocks.NAME2ID["planks"]
    world_obs[CURRENT_BLOCKS, 1, 2, 1] = planks
    obs_batch = world_obs[None], inventory_obs[None], timestep[None]

    mask = MbagActionDistribution.get_mask(env.config, obs_batch)

    # Can't do invalid actions.
    assert np.all(mask[0, MbagActionDistribution.MOVE_POS_X] == False)
    assert np.all(mask[0, MbagActionDistribution.GIVE_BLOCK] == False)

    # Can place a block next to other blocks.
    assert mask[0, MbagActionDistribution.PLACE_BLOCK][planks, 1, 2, 2] == True
    # Can't place a block where there is one.
    assert mask[0, MbagActionDistribution.PLACE_BLOCK][planks, 1, 1, 1] == False
    # Can't place a block floating in midair.
    assert mask[0, MbagActionDistribution.PLACE_BLOCK][planks, 1, 4, 1] == False

    # Can't place bedrock or air.
    assert (
        mask[0, MbagActionDistribution.PLACE_BLOCK][MinecraftBlocks.AIR, 1, 2, 2]
        == False
    )
    assert (
        mask[0, MbagActionDistribution.PLACE_BLOCK][MinecraftBlocks.BEDROCK, 1, 2, 2]
        == False
    )

    # Can't break bedrock or air.
    assert mask[0, MbagActionDistribution.BREAK_BLOCK, 1, 0, 1] == False
    assert mask[0, MbagActionDistribution.BREAK_BLOCK, 1, 2, 2] == False
    # Can break dirt and planks.
    assert mask[0, MbagActionDistribution.BREAK_BLOCK, 1, 1, 1] == True
    assert mask[0, MbagActionDistribution.BREAK_BLOCK, 1, 2, 1] == True


def test_mask_no_teleportation_no_inf_blocks():
    env = MbagEnv(
        {
            "world_size": (7, 7, 7),
            "num_players": 2,
            "players": [{}, {}],
            "abilities": {
                "teleportation": False,
                "flying": True,
                "inf_blocks": False,
            },
        }
    )
    world_obs, inventory_obs, timestep = env.reset()[0]

    # Suppose the player has dirt in their inventory.
    dirt = MinecraftBlocks.NAME2ID["dirt"]
    planks = MinecraftBlocks.NAME2ID["planks"]
    inventory_obs[dirt] += 1

    obs_batch = (
        world_obs[None].repeat(2, 0),
        inventory_obs[None].repeat(2, 0),
        timestep[None].repeat(2, 0),
    )
    mask = MbagActionDistribution.get_mask(env.config, obs_batch)

    # Player 1 should be at (0, 2, 0) and player 2 at (1, 2, 0).
    assert np.all(world_obs[PLAYER_LOCATIONS][0, 2:4, 0] == 1)
    assert np.all(world_obs[PLAYER_LOCATIONS][1, 2:4, 0] == 2)

    # They shouldn't be able to place/break blocks more than 3 blocks away.
    assert np.all(
        mask[:, MbagActionDistribution.PLACE_BLOCK][0, dirt, 5:, :, :] == False
    )
    assert np.all(
        mask[:, MbagActionDistribution.PLACE_BLOCK][0, dirt, :, :, 4:] == False
    )
    assert np.all(mask[0, MbagActionDistribution.BREAK_BLOCK, 5:, :, :] == False)
    assert np.all(mask[0, MbagActionDistribution.BREAK_BLOCK, :, :, 4:] == False)

    # Can only place blocks we have.
    assert mask[:, MbagActionDistribution.PLACE_BLOCK][0, dirt, 0, 2, 1] == True
    assert mask[:, MbagActionDistribution.PLACE_BLOCK][0, planks, 0, 2, 1] == False

    # Should only be able to give to a location where a player is.
    assert mask[:, MbagActionDistribution.GIVE_BLOCK][0, dirt, 1, 2, 0] == True
    assert mask[:, MbagActionDistribution.GIVE_BLOCK][0, dirt, 2, 2, 0] == False

    # Can only give blocks we have.
    assert mask[:, MbagActionDistribution.GIVE_BLOCK][0, planks, 1, 2, 0] == False

    world_obs, _, _ = env.step(
        [(MbagAction.NOOP, 0, 0), (MbagAction.MOVE_POS_X, 0, 0)]
    )[0][0]
    obs_batch = world_obs[None], inventory_obs[None], timestep[None]
    mask = MbagActionDistribution.get_mask(env.config, obs_batch)

    # Player 2 should now be at (2, 2, 0).
    assert np.all(world_obs[PLAYER_LOCATIONS][2, 2:4, 0] == 2)
    # Now it should be impossible to give blocks since player 2 is too far away.
    assert np.all(mask[0, MbagAction.GIVE_BLOCK] == False)


def test_to_flat():
    c = MbagActionDistribution.NUM_CHANNELS
    probs = np.ones((1, c, 2, 2, 2))
    probs /= probs.sum()

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["abilities"] = {
        "teleportation": True,
        "flying": True,
        "inf_blocks": True,
    }
    flat = MbagActionDistribution.to_flat(config, probs).flatten().tolist()
    flat_torch = (
        MbagActionDistribution.to_flat_torch(config, torch.from_numpy(probs))
        .flatten()
        .tolist()
    )
    flat_torch_logits = (
        MbagActionDistribution.to_flat_torch_logits(
            config, torch.from_numpy(probs).log()
        )
        .exp()
        .flatten()
        .tolist()
    )
    expected_flat = (
        [1 / c]  # NOOP
        + [1 / c / 8] * 8 * MinecraftBlocks.NUM_BLOCKS  # PLACE_BLOCK
        + [1 / c / 8] * 8  # BREAK_BLOCK
    )
    assert flat == pytest.approx(expected_flat)
    assert flat_torch == pytest.approx(expected_flat)
    assert flat_torch_logits == pytest.approx(expected_flat)

    config["abilities"] = {
        "teleportation": False,
        "flying": True,
        "inf_blocks": False,
    }
    flat = MbagActionDistribution.to_flat(config, probs).flatten().tolist()
    flat_torch = (
        MbagActionDistribution.to_flat_torch(config, torch.from_numpy(probs))
        .flatten()
        .tolist()
    )
    flat_torch_logits = (
        MbagActionDistribution.to_flat_torch_logits(
            config, torch.from_numpy(probs).log()
        )
        .exp()
        .flatten()
        .tolist()
    )
    expected_flat = (
        [1 / c]  # NOOP
        + [1 / c / 8] * 8 * MinecraftBlocks.NUM_BLOCKS  # PLACE_BLOCK
        + [1 / c / 8] * 8  # BREAK_BLOCK
        + [1 / c] * 6  # movement actions
        + [1 / c / 8] * 8 * MinecraftBlocks.NUM_BLOCKS  # GIVE_BLOCK
    )
    assert flat == pytest.approx(expected_flat)
    assert flat_torch == pytest.approx(expected_flat)
    assert flat_torch_logits == pytest.approx(expected_flat)
