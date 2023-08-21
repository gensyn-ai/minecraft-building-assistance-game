import logging

import pytest

from mbag.agents.hardcoded_agents import (
    HardcodedInventoryDonator,
    HardcodedInventoryReceiver,
    HardcodedResourceAgent,
)
from mbag.agents.heuristic_agents import NoopAgent
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.evaluation.evaluator import MbagEvaluator

logger = logging.getLogger(__name__)


def test_inventory():
    """
    Make sure the inventory agent can place blocks
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 20,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {"pallette": True},
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (
                HardcodedResourceAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 3


@pytest.mark.uses_malmo
def test_inventory_in_malmo():
    """
    Make sure the inventory agent can place blocks
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 20,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {"pallette": True},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (
                HardcodedResourceAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 3


def test_pallette():
    """
    Make sure the block pallette generates
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 1,
            "horizon": 10,
            "goal_generator": BasicGoalGenerator,
            "malmo": {
                "use_malmo": False,
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
    assert episode_info.cumulative_reward == 0


@pytest.mark.uses_malmo
def test_pallette_in_malmo():
    """
    Make sure the block pallette generates
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 1,
            "horizon": 10,
            "goal_generator": BasicGoalGenerator,
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (
                NoopAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 0


def test_give():
    """
    Make sure agents can give each other resources
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "players": [{}, {}],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (HardcodedInventoryDonator, {}),
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 0


@pytest.mark.uses_malmo
def test_give_in_malmo():
    """
    Make sure agents can give each other resources
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "players": [{}, {}],
            "malmo": {"use_malmo": True, "use_spectator": False, "video_dir": None},
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (HardcodedInventoryDonator, {}),
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 0


def test_give_with_moving_agents():
    """
    Make sure agents can give each other resources
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "players": [{}, {}],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (HardcodedInventoryDonator, {}),
            (HardcodedInventoryReceiver, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 5


@pytest.mark.uses_malmo
def test_give_with_moving_agents_in_malmo():
    """
    Make sure agents can give each other resources
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "players": [{}, {}],
            "malmo": {"use_malmo": True, "use_spectator": False, "video_dir": None},
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (HardcodedInventoryDonator, {}),
            (HardcodedInventoryReceiver, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0
