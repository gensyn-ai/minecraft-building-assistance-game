import glob
import json
import os
import tempfile
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals.goal_generator import GoalGenerator

import pytest
import numpy as np
from mbag.environment.types import WorldSize

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import NoopAgent, PriorityQueueAgent
from mbag.environment.goals.grabcraft import (
    GrabcraftGoalGenerator,
    CroppedGrabcraftGoalGenerator,
    SingleWallGrabcraftGenerator,
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
                PriorityQueueAgent,
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
            "goal_generator_config": {"data_dir": "data/grabcraft", "subset": "train"},
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


def test_crop_generator():
    config = {"data_dir": "data/grabcraft", "subset": "train"}
    world_size: WorldSize = (15, 10, 15)

    generator = CroppedGrabcraftGoalGenerator(config)
    goal = generator.generate_goal(world_size)
    assert goal.size == world_size

    generator = CroppedGrabcraftGoalGenerator(
        {**config, "force_single_cc": True, "use_limited_block_set": True}
    )
    goal = generator.generate_goal(world_size)
    assert goal.is_single_cc()

    evaluator = MbagEvaluator(
        {
            "world_size": (15, 10, 15),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": CroppedGrabcraftGoalGenerator,
            "goal_generator_config": config,
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
                "use_limited_block_set": True,
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


def test_generate_crop_json():
    data_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(data_dir, "hardcoded_crops"))
    os.symlink(
        os.path.abspath("data/grabcraft/train"),
        os.path.join(data_dir, "train"),
    )

    generator = CroppedGrabcraftGoalGenerator(
        {
            "data_dir": data_dir,
            "subset": "train",
            "force_single_cc": True,
            "use_limited_block_set": True,
            "save_crop_dir": "hardcoded_crops",
        }
    )

    for _ in range(10):
        structure = generator.generate_goal((8, 9, 8), save_crop=True)
        assert (
            structure.size[0] <= 8 and structure.size[1] <= 9 and structure.size[2] <= 8
        )

    for fname in glob.glob(os.path.join(data_dir, "hardcoded_crops", "*_crop.json")):
        with open(fname) as f:
            structure_json = json.load(f)

        size = generator._get_structure_size(structure_json)
        print(size)
        assert size[0] <= 8 and size[1] <= 9 and size[2] <= 8


def test_single_wall_generator():
    config = {"data_dir": "data/grabcraft", "subset": "train"}
    world_size: WorldSize = (10, 10, 15)

    generator = SingleWallGrabcraftGenerator(
        {
            **config,
            "use_limited_block_set": True,
            "min_density": 0.55,
        }
    )
    goal = generator.generate_goal(world_size)
    assert goal is not None

    evaluator = MbagEvaluator(
        {
            "world_size": world_size,
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": SingleWallGrabcraftGenerator,
            "goal_generator_config": config,
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
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_single_wall_generator_with_alternate_settings():
    config = {"data_dir": "data/grabcraft", "subset": "train"}
    world_size: WorldSize = (10, 10, 15)

    generator = SingleWallGrabcraftGenerator(
        {
            **config,
            "use_limited_block_set": True,
            "choose_densest": True,
            "make_symmetric": False,
            "force_single_cc": True,
        }
    )
    goal = generator.generate_goal(world_size)
    assert goal.is_single_cc()

    evaluator = MbagEvaluator(
        {
            "world_size": world_size,
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": SingleWallGrabcraftGenerator,
            "goal_generator_config": config,
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
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


@pytest.mark.xfail(strict=False)
def test_single_wall_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": SingleWallGrabcraftGenerator,
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
                "use_limited_block_set": True,
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
                NoopAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_single_wall_generator_hard_coded_crop():
    size = (10, 10, 15)
    config = {"data_dir": "data/grabcraft", "subset": "train"}

    generator = SingleWallGrabcraftGenerator(
        {
            **config,
            "use_limited_block_set": True,
            "choose_densest": True,
            "make_symmetric": False,
            "force_single_cc": True,
        }
    )

    structure = generator._get_structure("5861")
    assert structure is not None

    crop = generator._generate_wall_crop(size, structure)

    assert crop is not None
    assert crop.is_single_cc()


# This takes about 85 seconds to run, so it will fail the 10 second timeout.
@pytest.mark.xfail(strict=False)
def test_get_sample_size():
    size = (10, 10, 15)
    generator = SingleWallGrabcraftGenerator(
        {"data_dir": "data/grabcraft", "subset": "train"}
    )

    result = generator.get_sample_size(size)

    assert result is not None
    assert result == 32708

    generator = SingleWallGrabcraftGenerator(
        {"data_dir": "data/grabcraft", "subset": "test"}
    )

    result = generator.get_sample_size(size)

    assert result is not None
    assert result == 4938


@pytest.mark.xfail(strict=False)
def test_make_uniform_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": SingleWallGrabcraftGenerator,
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
                "use_limited_block_set": True,
                "make_uniform": True,
                "uniform_block": MinecraftBlocks.NAME2ID["cobblestone"],
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
                NoopAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_make_uniform():
    size = (10, 10, 15)
    config = {"data_dir": "data/grabcraft", "subset": "train"}

    grass_block = MinecraftBlocks.NAME2ID["grass"]
    cobblestone_block = MinecraftBlocks.NAME2ID["cobblestone"]

    generator = SingleWallGrabcraftGenerator(
        {
            **config,
            "use_limited_block_set": True,
            "choose_densest": True,
            "make_symmetric": False,
            "force_single_cc": True,
        }
    )

    structure = generator._get_structure("5861")
    assert structure is not None
    crop = generator._generate_wall_crop(size, structure)
    assert crop is not None
    uniform = GoalGenerator.make_uniform(crop, cobblestone_block)
    crop_blocks, uniform_blocks = crop.blocks, uniform.blocks

    assert uniform_blocks is not None
    assert np.all(
        (crop.blocks == MinecraftBlocks.AIR) == (uniform_blocks == MinecraftBlocks.AIR)
    )
    assert np.all(
        (crop_blocks[:, 0, :] == grass_block)
        == (uniform_blocks[:, 0, :] == grass_block)
    )
    assert np.all(
        (crop_blocks[:, 0, :] == MinecraftBlocks.AIR)
        == (uniform_blocks[:, 0, :] == MinecraftBlocks.AIR)
    )
    assert np.all(
        (
            (crop_blocks[:, 0, :] != MinecraftBlocks.AIR)
            & (crop_blocks[:, 0, :] != grass_block)
        )
        == (uniform_blocks[:, 0, :] == cobblestone_block)
    )
    assert np.all(
        uniform_blocks[:, 1:, :][crop_blocks[:, 1:, :] != MinecraftBlocks.AIR]
        == cobblestone_block
    )
    assert np.all(uniform_blocks[uniform_blocks != crop_blocks] == cobblestone_block)
