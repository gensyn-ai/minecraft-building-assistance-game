from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent
from mbag.environment.goals.simple import (
    BasicGoalGenerator,
)


def test_layer_builder_agent():
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
        force_get_set_state=True,
    )
    reward = evaluator.rollout()
    assert reward == 18
