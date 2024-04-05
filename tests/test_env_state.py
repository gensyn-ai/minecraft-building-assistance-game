import random
from typing import cast

import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagActionTuple
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.state import mbag_obs_to_state


def test_obs_to_state():
    for num_players in [1, 2]:
        for teleportation in [False, True]:
            for inf_blocks in [False, True]:
                horizon = 100
                env = MbagEnv(
                    {
                        "num_players": num_players,
                        "players": [{} for _ in range(num_players)],
                        "goal_generator": "random",
                        "horizon": horizon,
                        "terminate_on_goal_completion": False,
                        "abilities": {
                            "teleportation": teleportation,
                            "flying": True,
                            "inf_blocks": inf_blocks,
                        },
                        "world_size": (5, 5, 5),
                    }
                )
                obs, _ = env.reset()
                action_map = MbagActionDistribution.get_action_mapping(env.config)

                for t in range(horizon):
                    player_index = random.randrange(num_players)
                    state = mbag_obs_to_state(obs[player_index], player_index)
                    new_obs = env.set_state(state)

                    for expected_player_obs, player_obs in zip(obs, new_obs):
                        for expected_obs_piece, obs_piece in zip(
                            expected_player_obs, player_obs
                        ):
                            np.testing.assert_array_equal(expected_obs_piece, obs_piece)

                    actions = []

                    for player_obs in obs:
                        obs_batch = (
                            player_obs[0][None],
                            player_obs[1][None],
                            player_obs[2][None],
                        )
                        mask = MbagActionDistribution.get_mask_flat(
                            env.config, obs_batch
                        )[0]
                        flat_action = random.choice(mask.nonzero()[0])
                        action = cast(MbagActionTuple, tuple(action_map[flat_action]))
                        actions.append(action)

                    obs, _, _, _ = env.step(actions)
