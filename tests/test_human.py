import logging

import pytest

from mbag.agents.human_agent import HumanAgent
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import NoopAgent

logger = logging.getLogger(__name__)


@pytest.mark.uses_malmo
@pytest.mark.timeout(3600)
def test_human_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {"pallette": True},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
                "restrict_players": True,
            },
            "players": [{"is_human": True}],
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (HumanAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 0


@pytest.mark.uses_malmo
@pytest.mark.timeout(3600)
def test_two_humans_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 2,
            "horizon": 1000,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {"pallette": True},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
                "restrict_players": True,
            },
            "players": [
                {"is_human": True},
                {"is_human": True},
            ],
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (HumanAgent, {}),
            (HumanAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 0


@pytest.mark.uses_malmo
@pytest.mark.timeout(3600)
def test_with_items_one_player():
    """
    Make sure you can initialize players with resources
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "players": [
                {
                    "give_items": [
                        {
                            "id": "diamond_pickaxe",
                            "count": 1,
                            "enchantments": [
                                {
                                    "id": 33,  # gives silk touch enchantment, lvl defaults to max lvl
                                },
                                {
                                    "id": 34,  # gives unbreaking enchantment
                                    "level": 3,  # manually set the level
                                },
                            ],
                        }
                    ]
                }
            ],
            "malmo": {"use_malmo": True, "use_spectator": False, "video_dir": None},
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 0


@pytest.mark.uses_malmo
@pytest.mark.timeout(3600)
def test_with_items_two_player():
    """
    Make sure you can initialize players with resources
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "players": [
                {
                    "give_items": [
                        {
                            "id": "diamond_pickaxe",
                            "count": 1,
                            "enchantments": [
                                {
                                    "id": 33,  # gives silk touch enchantment, lvl defaults to max lvl
                                },
                                {
                                    "id": 34,  # gives unbreaking enchantment
                                    "level": 3,  # manually set the level
                                },
                            ],
                        }
                    ]
                },
                {
                    "give_items": [
                        {
                            "id": "diamond_pickaxe",
                            "count": 1,
                            "enchantments": [
                                {
                                    "id": 33,  # gives silk touch enchantment, lvl defaults to max lvl
                                },
                            ],
                        }
                    ]
                },
            ],
            "malmo": {"use_malmo": True, "use_spectator": False, "video_dir": None},
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (NoopAgent, {}),
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 0
