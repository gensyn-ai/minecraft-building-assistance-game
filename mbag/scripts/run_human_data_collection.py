import logging
import os
import pickle
from datetime import datetime
from subprocess import Popen
from typing import Optional

from malmo import minecraft
from sacred import Experiment
from sacred.observers import FileStorageObserver

from mbag.agents.human_agent import HumanAgent
from mbag.environment.config import DEFAULT_HUMAN_GIVE_ITEMS
from mbag.environment.goals import TransformedGoalGenerator, TutorialGoalGenerator
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import WorldSize
from mbag.evaluation.evaluator import MbagEvaluator

logger = logging.getLogger(__name__)

ex = Experiment()


@ex.config
def make_human_action_config():
    launch_minecraft = False  # noqa: F841
    data_path = "data/human_data"  # noqa: F841

    num_players = 2
    world_size: WorldSize = (11, 10, 10)
    goal_generator = TransformedGoalGenerator
    house_id = None
    goal_generator_config = {
        "goal_generator": "craftassist",
        "goal_generator_config": {
            "data_dir": "data/craftassist",
            "subset": "train",
            "house_id": house_id,
        },
        "transforms": [
            {
                "transform": "largest_cc",
                "config": {"connectivity": 18},
            },
            {"transform": "crop_air"},
            {
                "transform": "crop_low_density_bottom_layers",
                "config": {"density_threshold": 0.1},
            },
            {
                "transform": "min_size_filter",
                "config": {"min_size": [4, 4, 4]},
            },
            {
                "transform": "area_sample",
                "config": {
                    "interpolate": True,
                    "interpolation_order": 1,
                    "max_scaling_factor": 2,
                    "max_scaling_factor_ratio": 1.5,
                    "preserve_paths": True,
                    "scale_y_independently": True,
                },
            },
            {
                "transform": "density_filter",
                "config": {"max_density": 1, "min_density": 0},
            },
            {"transform": "randomly_place"},
            {"transform": "add_grass"},
            {
                "transform": "single_cc_filter",
                "config": {"connectivity": 18},
            },
        ],
    }

    mbag_config: MbagConfigDict = {  # noqa: F841
        "world_size": world_size,
        "num_players": num_players,
        "horizon": 1_000_000_000,
        "goal_generator": goal_generator,
        "goal_generator_config": goal_generator_config,
        "malmo": {
            "use_malmo": True,
            "use_spectator": False,
            "video_dir": None,
            "restrict_players": True,
            "ssh_args": [None for _ in range(num_players)],
            "start_port": 10000,
            "action_delay": 0.001,
        },
        "players": [
            {
                "is_human": True,
                "give_items": DEFAULT_HUMAN_GIVE_ITEMS,
            }
            for _ in range(num_players)
        ],
        "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
    }

    experiment_parts = []  # type: ignore
    if house_id is not None:
        experiment_parts.append(house_id)
    result_dir = os.path.join(
        data_path, *experiment_parts, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    observer = FileStorageObserver(result_dir)
    ex.observers.append(observer)


@ex.named_config
def tutorial():
    world_size = (6, 6, 6)  # noqa: F841
    goal_generator = TutorialGoalGenerator  # noqa: F841
    goal_generator_config = {}  # noqa: F841
    num_players = 1  # noqa: F841


@ex.automain
def main(
    launch_minecraft: bool,
    num_players: int,
    mbag_config: MbagConfigDict,
    observer: FileStorageObserver,
):
    minecraft_process: Optional[Popen] = None
    if launch_minecraft:
        (minecraft_process,) = minecraft.launch()

    evaluator = MbagEvaluator(
        mbag_config,
        [(HumanAgent, {}) for _ in range(num_players)],
        return_on_exception=True,
    )

    episode_info = evaluator.rollout()
    assert observer.dir is not None
    episode_fname = os.path.join(observer.dir, "episode.pkl")
    logger.info(f"saving episode to {episode_fname}")
    with open(episode_fname, "wb") as episode_file:
        pickle.dump(episode_info, episode_file)

    if minecraft_process is not None:
        minecraft_process.terminate()
