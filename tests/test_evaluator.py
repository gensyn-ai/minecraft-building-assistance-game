import pytest

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent, NoopAgent


def test_single_agent():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 50,
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
    assert reward == 25


def test_two_agents():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_visibility": [True, True],
            "use_malmo": False,
        },
        [
            (LayerBuilderAgent, {}),
            (NoopAgent, {}),
        ],
    )
    reward = evaluator.rollout()
    assert reward == 25


@pytest.mark.xfail(strict=False)
def test_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 100,
            "goal_visibility": [True],
            "use_malmo": True,
        },
        [
            (LayerBuilderAgent, {}),
        ],
    )
    reward = evaluator.rollout()
    assert reward == 25
