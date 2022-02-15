import pytest

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent
from mbag.environment.goals.grabcraft import (
    GrabcraftGoalGenerator,
    CroppedGrabcraftGoalGenerator,
)


def test_goal_generator():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 500,
            "goal_generator": GrabcraftGoalGenerator,
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
            },
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
def test_goal_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": GrabcraftGoalGenerator,
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
            },
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


@pytest.mark.xfail(strict=False)
def test_crop_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (15, 10, 15),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": CroppedGrabcraftGoalGenerator,
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
            },
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
