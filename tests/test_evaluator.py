import pytest

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent, NoopAgent
from mbag.environment.goals.simple import BasicGoalGenerator


def test_single_agent():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 50,
            "goal_generator": (BasicGoalGenerator, {}),
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
    assert reward == 13.5


def test_two_agents():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": (BasicGoalGenerator, {}),
            "goal_visibility": [True, True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (LayerBuilderAgent, {}),
            (NoopAgent, {}),
        ],
    )
    reward = evaluator.rollout()
    assert reward == 13.5


@pytest.mark.xfail(strict=False)
def test_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 100,
            "goal_generator": (BasicGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (LayerBuilderAgent, {}),
        ],
    )
    reward = evaluator.rollout()
    assert reward == 9
