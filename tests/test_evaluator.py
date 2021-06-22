from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent


def test_single_agent():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 50,
            "goal_visibility": [True],
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
