import logging
import random
from typing import List

import numpy as np

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.types import MbagAction, WorldLocation

logger = logging.getLogger(__name__)


def test_place_break_through_player():
    """
    Make sure we can't place/break blocks through another player.
    """

    blocks = MinecraftBlocks((3, 3, 3))
    dirt = MinecraftBlocks.NAME2ID["dirt"]
    blocks.blocks[:, 0, :] = dirt
    player_location: WorldLocation = (0.5, 1, 1.5)
    assert (
        blocks.try_break_place(
            MbagAction.PLACE_BLOCK,
            (2, 1, 1),
            dirt,
            player_location=player_location,
            update_blocks=False,
        )
        is not None
    )

    other_player_locations: List[WorldLocation] = [(1.5, 1, 1.5)]
    assert (
        blocks.try_break_place(
            MbagAction.PLACE_BLOCK,
            (2, 1, 1),
            dirt,
            player_location=player_location,
            other_player_locations=other_player_locations,
            update_blocks=False,
        )
        is None
    )
    assert (
        blocks.try_break_place(
            MbagAction.BREAK_BLOCK,
            (2, 0, 1),
            player_location=player_location,
            other_player_locations=other_player_locations,
            update_blocks=False,
        )
        is None
    )


def test_get_viewpoint_click_candidates_c_extension():
    import _mbag  # noqa: F401

    for teleportation in [True, False]:
        for num_players in [1, 2, 3]:
            logger.info(
                f"testing teleportation={teleportation}, num_players={num_players}"
            )
            env = MbagEnv(
                {
                    "abilities": {
                        "teleportation": teleportation,
                        "inf_blocks": True,
                        "flying": True,
                    },
                    "num_players": num_players,
                    "random_start_locations": True,
                    "players": [{} for _ in range(num_players)],
                }
            )
            width, height, depth = env.config["world_size"]
            for _ in range(10):
                env.reset(force_regenerate_goal=True)
                if teleportation:
                    player_location = None
                else:
                    player_location = env.player_locations[0]
                block_location = (
                    random.randrange(width),
                    random.randrange(height),
                    random.randrange(depth),
                )
                action_type = random.choice(
                    [MbagAction.PLACE_BLOCK, MbagAction.BREAK_BLOCK]
                )
                candidates_python = env.current_blocks._get_viewpoint_click_candidates(
                    action_type,
                    block_location,
                    player_location,
                    env.player_locations[1:],
                    force_python_impl=True,
                )
                candidates_c = env.current_blocks._get_viewpoint_click_candidates(
                    action_type,
                    block_location,
                    player_location,
                    env.player_locations[1:],
                )
                np.testing.assert_allclose(candidates_python, candidates_c)
