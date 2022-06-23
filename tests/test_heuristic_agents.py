from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import MbagAction
import pytest
import numpy as np
import logging
from numpy.testing import assert_array_equal

from mbag.environment.goals.grabcraft import (
    GrabcraftGoalGenerator,
    SingleWallGrabcraftGenerator,
)
from mbag.environment.mbag_env import MbagConfigDict
from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import (
    LayerBuilderAgent,
    PriorityQueueAgent,
    MirrorBuildingAgent,
    ALL_HEURISTIC_AGENTS,
)
from mbag.environment.goals.simple import (
    BasicGoalGenerator,
    SimpleOverhangGoalGenerator,
)

logger = logging.getLogger(__name__)


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
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 18


def test_pq_agent_basic():
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
                PriorityQueueAgent,
                {},
            )
        ],
        force_get_set_state=True,
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 18


def test_pq_agent_overhangs():
    for num_players in [1, 2]:
        evaluator = MbagEvaluator(
            {
                "world_size": (8, 8, 8),
                "num_players": num_players,
                "horizon": 100,
                "goal_generator": (SimpleOverhangGoalGenerator, {}),
                "goal_visibility": [True] * num_players,
                "malmo": {
                    "use_malmo": False,
                    "use_spectator": False,
                    "video_dir": None,
                },
            },
            [
                (PriorityQueueAgent, {}),
            ]
            * num_players,
            force_get_set_state=True,
        )
        episode_info = evaluator.rollout()
        assert episode_info.cumulative_reward == 13


def test_pq_agent_grabcraft():
    for num_players in [1, 2]:
        evaluator = MbagEvaluator(
            {
                "world_size": (12, 12, 12),
                "num_players": num_players,
                "horizon": 1000,
                "goal_generator": GrabcraftGoalGenerator,
                "goal_generator_config": {
                    "data_dir": "data/grabcraft",
                    "subset": "train",
                    "force_single_cc": True,
                },
                "goal_visibility": [True] * num_players,
                "malmo": {
                    "use_malmo": False,
                    "use_spectator": False,
                    "video_dir": None,
                },
            },
            [
                (PriorityQueueAgent, {}),
            ]
            * num_players,
        )
        episode_info = evaluator.rollout()
        (last_obs,) = episode_info.last_obs[0]
        if not np.all(last_obs[0] == last_obs[2]):
            for layer in range(12):
                if not np.all(last_obs[0, :, layer] == last_obs[2, :, layer]):
                    print(f"Mismatch in layer {layer}")
                    print("Current blocks:")
                    print(last_obs[0, :, layer])
                    print("Goal blocks:")
                    print(last_obs[2, :, layer])
                    break
        assert_array_equal(
            last_obs[0], last_obs[2], "Agent should finish building house."
        )


@pytest.mark.xfail(strict=False)
def test_malmo_pq():
    evaluator = MbagEvaluator(
        {
            "world_size": (8, 8, 8),
            "num_players": 1,
            "horizon": 100,
            "goal_generator": (SimpleOverhangGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (PriorityQueueAgent, {}),
        ],
        force_get_set_state=True,
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 13


def test_rllib_heuristic_agents():
    from ray.rllib.agents.pg.pg import PGTrainer
    from ray.rllib.rollout import rollout
    from mbag.rllib.policies import MbagAgentPolicy

    env_config: MbagConfigDict = {
        "world_size": (8, 8, 8),
        "num_players": 1,
        "horizon": 100,
        "goal_generator": (BasicGoalGenerator, {}),
        "goal_visibility": [True],
        "malmo": {
            "use_malmo": False,
            "use_spectator": False,
            "video_dir": None,
        },
    }

    agents = ALL_HEURISTIC_AGENTS.copy()
    del agents["mirror_builder"]
    for heuristic_agent_id, heuristic_agent_cls in agents.items():
        logger.info(f"Testing {heuristic_agent_id} agent...")
        heuristic_agent = heuristic_agent_cls({}, env_config)
        trainer = PGTrainer(
            {
                "env": "MBAG-v1",
                "env_config": env_config,
                "multiagent": {
                    "policies": {
                        "pq": (
                            MbagAgentPolicy,
                            None,
                            None,
                            {"mbag_agent": heuristic_agent},
                        )
                    },
                    "policy_mapping_fn": lambda agent_id: "pq",
                    "policies_to_train": [],
                },
                "framework": "torch",
            }
        )

        rollout(
            trainer,
            None,
            num_steps=0,
            num_episodes=2,
        )


def test_mirror_x_index():
    agent = MirrorBuildingAgent({"world_size": (10, 10, 10)}, {})

    K = 10
    for i in range(K):
        assert (agent._mirror_x_index(np.array([i]), K) == np.array([K - i - 1])).all()
        assert (
            agent._mirror_x_index(np.array([i, 3, 5]), K) == np.array([K - i - 1, 3, 5])
        ).all()

    K = 9
    for i in range(K):
        assert (
            agent._mirror_x_index(np.array([i, 3, 5]), K) == np.array([K - i - 1, 3, 5])
        ).all()


def test_diff_indices():
    agent = MirrorBuildingAgent({"world_size": (10, 10, 10)}, {})

    a = np.zeros((5, 5, 5))
    b = np.ones((5, 5, 5))
    assert len(agent._diff_indices(a, a)) == 0

    diffs = agent._diff_indices(
        a,
        b,
    )
    assert len(diffs) == 5**3
    for x in range(5):
        for y in range(5):
            for z in range(5):
                assert np.array([x, y, z]) in diffs

    c = a.copy()
    c[0, 0, 0] = 1
    assert np.array_equal(agent._diff_indices(a, c), [[0, 0, 0]])
    c[1, 2, 3] = 1
    assert np.array_equal(agent._diff_indices(a, c), [[0, 0, 0], [1, 2, 3]])


def test_mirror_building_agent_get_action():
    agent = MirrorBuildingAgent({}, {"world_size": (3, 3, 3)})

    dim = (3, 3, 3, 3)
    a = np.zeros(dim)
    a[
        0,
    ] = MinecraftBlocks.AIR

    assert agent.get_action((a,)) == (MbagAction.NOOP, 0, 0)
    assert agent.get_action((a,)) == (MbagAction.NOOP, 0, 0)

    b = a.copy()
    DIRT = MinecraftBlocks.NAME2ID["dirt"]
    b[0, 0, 0, 0] = DIRT

    assert str(agent.get_action((b,))) == str((MbagAction.PLACE_BLOCK, 18, DIRT))
    assert str(agent.get_action((a,))) == str((MbagAction.NOOP, 0, 0))

    c = b.copy()
    c[0, 2, 0, 0] = DIRT

    agent.get_action((c,))
    assert str(agent.get_action((b,))) == str((MbagAction.BREAK_BLOCK, 0, 0))


def test_mirror_building_agent():
    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": (
                SingleWallGrabcraftGenerator,
                {"test_wall": True, "choose_densest": True},
            ),
            "goal_visibility": [True, False],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [(LayerBuilderAgent, {}), (MirrorBuildingAgent, {})],
        force_get_set_state=False,
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 44


@pytest.mark.xfail(strict=False)
def test_mirror_building_agent_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 2,
            "horizon": 100,
            "goal_generator": (SingleWallGrabcraftGenerator, {}),
            "goal_visibility": [True, False],
            "malmo": {
                "use_malmo": True,
                "use_spectator": True,
                "video_dir": None,
            },
        },
        [(LayerBuilderAgent, {}), (MirrorBuildingAgent, {})],
        force_get_set_state=False,
    )
    episode_info = evaluator.rollout()
