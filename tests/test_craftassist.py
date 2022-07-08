from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import PriorityQueueAgent


def test_goal_generator():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 500,
            "goal_generator_config": {
                "goal_generator": "craftassist",
                "goal_generator_config": {
                    "data_dir": "data/craftassist",
                    "train": True,
                },
                "goal_transforms": [
                    {"transform": "randomly_place"},
                ],
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
                PriorityQueueAgent,
                {},
            )
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_goal_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator_config": {
                "goal_generator": "craftassist",
                "goal_generator_config": {
                    "data_dir": "data/craftassist",
                    "train": True,
                },
                "goal_transforms": [
                    {"transform": "randomly_place"},
                ],
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
                PriorityQueueAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0
