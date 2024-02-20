import pytest

from mbag.agents.hardcoded_agents import HardcodedBuilderAgent
from mbag.agents.heuristic_agents import LayerBuilderAgent, MovementAgent, NoopAgent
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.types import PLAYER_LOCATIONS
from mbag.evaluation.evaluator import MbagEvaluator


def test_movement():
    evaluator = MbagEvaluator(
        {
            "world_size": (12, 12, 12),
            "num_players": 1,
            "horizon": 10,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": True},
        },
        [
            (
                MovementAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 0


def test_random_start_locations():
    for num_players in [1, 2, 3]:
        env = MbagEnv(
            {
                # Make the world deliberately small so there is a higher chance of
                # collisions if we did something wrong.
                "world_size": (3, 3, 3),
                "num_players": num_players,
                "horizon": 10,
                "random_start_locations": True,
                "goal_generator": BasicGoalGenerator,
                "abilities": {
                    "teleportation": False,
                    "flying": True,
                    "inf_blocks": True,
                },
                "players": [{} for _ in range(num_players)],
            },
        )

        all_location_obs = set()
        for episode_index in range(5):
            obs_list, _ = env.reset(force_regenerate_goal=True)
            (
                world_obs,
                _,
                _,
            ) = obs_list[0]
            location_obs = world_obs[PLAYER_LOCATIONS]
            all_location_obs.add(location_obs.tobytes())
        assert len(all_location_obs) > 1


@pytest.mark.uses_malmo
def test_movement_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (12, 12, 12),
            "num_players": 1,
            "horizon": 10,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": True},
        },
        [
            (
                MovementAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == -1


def test_movement_with_building():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 30,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": True},
        },
        [
            (
                HardcodedBuilderAgent,
                {},
            )
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 9


@pytest.mark.uses_malmo
def test_movement_with_building_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 30,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": True},
        },
        [
            (
                HardcodedBuilderAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 9


def test_obstructing_agents():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 10,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "players": [{} for _ in range(10)],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"flying": True, "teleportation": False, "inf_blocks": True},
        },
        [
            (LayerBuilderAgent, {}),
            (NoopAgent, {}),
            (NoopAgent, {}),
            (NoopAgent, {}),
            (NoopAgent, {}),
            (NoopAgent, {}),
            (NoopAgent, {}),
            (NoopAgent, {}),
            (NoopAgent, {}),
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward < 18
