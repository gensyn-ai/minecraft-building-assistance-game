import pytest

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent
from mbag.environment.goals.grabcraft import GrabcraftGoalGenerator


def test_goal_generator():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 500,
            "goal_generator": (GrabcraftGoalGenerator, {"data_dir": "data/grabcraft"}),
            "goal_visibility": [True],
            "use_malmo": False,
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
            "goal_generator": (GrabcraftGoalGenerator, {"data_dir": "data/grabcraft"}),
            "goal_visibility": [True],
            "use_malmo": True,
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
