import pytest

from mbag.agents.heuristic_agents import PriorityQueueAgent
from mbag.environment.goals.goal_transform import TransformedGoalGenerator
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
def test_area_sampling_filter():
    area_sampling_generator = TransformedGoalGenerator(
        {
            "goal_generator": "grabcraft",
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
            },
            "transforms": [
                {
                    "transform": "area_sample",
                    "config": {"max_scaling_factor": 3, "interpolate": True},
                }
            ],
        }
    )

    structure = area_sampling_generator.generate_goal((15, 15, 15))

    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": TransformedGoalGenerator,
            "goal_generator_config": {
                "goal_generator": "from_minecraft_blocks",
                "goal_generator_config": {"structure": structure},
                "transforms": [
                    {
                        "transform": "seam_carve",
                        "config": {},
                    }
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
