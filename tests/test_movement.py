import pytest

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import MovementAgent, HardcodedBuilderAgent
from mbag.environment.goals.simple import (
    BasicGoalGenerator,
)


def test_movement():
    evaluator = MbagEvaluator(
        {
            "world_size": (12, 12, 12),
            "num_players": 1,
            "horizon": 10,
            "goal_generator": (BasicGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False},
        },
        [
            (
                MovementAgent,
                {},
            ),
        ],
    )
    reward = evaluator.rollout()
    assert reward == 0


@pytest.mark.xfail(strict=False)
def test_movement_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (12, 12, 12),
            "num_players": 1,
            "horizon": 10,
            "goal_generator": (BasicGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True},
        },
        [
            (
                MovementAgent,
                {},
            ),
        ],
    )
    reward = evaluator.rollout()
    assert reward == -1


def test_movement_with_building():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 30,
            "goal_generator": (BasicGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True},
        },
        [
            (
                HardcodedBuilderAgent,
                {},
            )
        ],
    )
    reward = evaluator.rollout()
    assert reward == 9


@pytest.mark.xfail(strict=False)
def test_movement_with_building_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 30,
            "goal_generator": (BasicGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True},
        },
        [
            (
                HardcodedBuilderAgent,
                {},
            ),
        ],
    )
    reward = evaluator.rollout()
    assert reward > 0
