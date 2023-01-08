import pytest

from mbag.agents.hardcoded_agents import HardcodedBuilderAgent
from mbag.agents.heuristic_agents import LayerBuilderAgent, MovementAgent, NoopAgent
from mbag.environment.goals.simple import BasicGoalGenerator
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
