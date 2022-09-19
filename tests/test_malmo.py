import tempfile
import os
import pytest

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import LayerBuilderAgent, NoopAgent
from mbag.environment.goals.simple import BasicGoalGenerator


@pytest.mark.uses_malmo
def test_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 100,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
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
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 18


@pytest.mark.uses_malmo
def test_two_agents_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "players": [{}, {}],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (LayerBuilderAgent, {}),
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 18


@pytest.mark.uses_malmo
def test_video_dir():
    with tempfile.TemporaryDirectory() as video_dir:
        evaluator = MbagEvaluator(
            {
                "world_size": (5, 5, 5),
                "num_players": 1,
                "horizon": 100,
                "goal_generator": BasicGoalGenerator,
                "goal_generator_config": {},
                "malmo": {
                    "use_malmo": True,
                    "use_spectator": True,
                    "video_dir": video_dir,
                },
            },
            [
                (LayerBuilderAgent, {}),
            ],
        )
        evaluator.rollout()
        assert os.path.exists(os.path.join(video_dir, "000000.mp4"))


@pytest.mark.uses_malmo
def test_human_collection():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "players": [
                {
                    "player_name": "human",
                    "goal_visible": True,
                    "is_human": True,
                    "timestep_skip": 1,
                    "rewards": {},
                    "give_items": [{"id": "diamond_pickaxe", "count": 1}],
                }
            ],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
                "restrict_players": True,
            },
            "abilities": {"inf_blocks": False, "teleportation": True, "flying": True},
        },
        [(NoopAgent, {})],
    )
    evaluator.rollout()
