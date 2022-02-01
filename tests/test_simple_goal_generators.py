import pytest

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent
from mbag.environment.goals.simple import (
    RandomGoalGenerator,
    SimpleOverhangGoalGenerator,
)


def test_random_goal_generator():
    evaluator = MbagEvaluator(
        {
            "world_size": (8, 8, 8),
            "num_players": 1,
            "horizon": 250,
            "goal_generator": (RandomGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                LayerBuilderAgent,
                {},
            )
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


@pytest.mark.xfail(strict=False)
def test_random_goal_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (8, 8, 8),
            "num_players": 1,
            "horizon": 250,
            "goal_generator": (RandomGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                LayerBuilderAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_simple_overhang_goal_generator():
    evaluator = MbagEvaluator(
        {
            "world_size": (8, 8, 8),
            "num_players": 1,
            "horizon": 250,
            "goal_generator": (SimpleOverhangGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                LayerBuilderAgent,
                {},
            )
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


@pytest.mark.xfail(strict=False)
def test_simple_overhang_goal_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (8, 8, 8),
            "num_players": 1,
            "horizon": 250,
            "goal_generator": (SimpleOverhangGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                LayerBuilderAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0
