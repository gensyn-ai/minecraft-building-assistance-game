import pytest

from mbag.agents.heuristic_agents import NoopAgent, PriorityQueueAgent
from mbag.environment.goals.goal_transform import TransformedGoalGenerator
from mbag.environment.goals.grabcraft import (
    GrabcraftGoalGenerator,
    ScaledDownGrabcraftGoalGenerator,
)
from mbag.evaluation.evaluator import MbagEvaluator


def test_goal_generator():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 500,
            "goal_generator": TransformedGoalGenerator,
            "goal_generator_config": {
                "goal_generator": "grabcraft",
                "goal_generator_config": {
                    "data_dir": "data/grabcraft",
                    "subset": "train",
                },
                "transforms": [
                    {"transform": "randomly_place"},
                ],
            },
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            )
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


@pytest.mark.uses_malmo
def test_goal_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": TransformedGoalGenerator,
            "goal_generator_config": {
                "goal_generator": "grabcraft",
                "goal_generator_config": {
                    "data_dir": "data/grabcraft",
                    "subset": "train",
                },
                "transforms": [
                    {"transform": "randomly_place"},
                ],
            },
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


@pytest.mark.uses_malmo
def test_get_full_grabcraft_structure():
    evaluator = MbagEvaluator(
        {
            "world_size": (25, 25, 25),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": GrabcraftGoalGenerator,
            "goal_generator_config": {"data_dir": "data/grabcraft", "subset": "train"},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                NoopAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


@pytest.mark.uses_malmo
def test_get_scaled_down_grabcraft_structure():
    evaluator = MbagEvaluator(
        {
            "world_size": (25, 25, 25),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": ScaledDownGrabcraftGoalGenerator,
            "goal_generator_config": {"data_dir": "data/grabcraft", "subset": "train"},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                NoopAgent,
                {},
            ),
        ],
    )

    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_scale_down_generator():
    generator = ScaledDownGrabcraftGoalGenerator({})
    result = generator.generate_goal((5, 5, 5))

    print(result.blocks)
