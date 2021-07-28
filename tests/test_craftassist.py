import pytest

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent
from mbag.environment.goals import CraftAssistGoalGenerator


def test_goal_generator():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 500,
            "goal_generator": (
                CraftAssistGoalGenerator,
                {
                    "data_dir": "data/craftassist",
                    "train": True,
                },
            ),
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
    reward = evaluator.rollout()
    assert reward > 0


@pytest.mark.xfail(strict=False)
def test_goal_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": (
                CraftAssistGoalGenerator,
                {
                    "data_dir": "data/craftassist",
                    "train": True,
                },
            ),
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
    reward = evaluator.rollout()
    assert reward > 0
