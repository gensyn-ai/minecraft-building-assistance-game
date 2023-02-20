import json
import logging
import os
from datetime import datetime
from subprocess import Popen
from typing import Optional

from malmo import minecraft
from ray.tune.utils.util import SafeFallbackEncoder
from sacred import Experiment

from mbag.agents.human_agent import HumanAgent
from mbag.environment.goals import TransformedGoalGenerator
from mbag.environment.mbag_env import MbagConfigDict
from mbag.evaluation.evaluator import MbagEvaluator

logger = logging.getLogger(__name__)

ex = Experiment()


@ex.config
def make_human_action_config():
    launch_minecraft = False  # noqa: F841
    data_path = "data/human_data"  # noqa: F841
    horizon = 10000

    num_players = 2

    mbag_config: MbagConfigDict = {  # noqa: F841
        "world_size": (10, 10, 10),
        "num_players": num_players,
        "horizon": horizon,
        "goal_generator": TransformedGoalGenerator,
        "goal_generator_config": {
            "goal_generator": "grabcraft",
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
            },
            "transforms": [
                {
                    "transform": "crop",
                    "config": {
                        "density_threshold": 0.25,
                        "tethered_to_ground": True,
                        "wall": False,
                    },
                },
                {"transform": "single_cc_filter"},
                {"transform": "randomly_place"},
                {"transform": "add_grass"},
            ],
        },
        "malmo": {
            "use_malmo": True,
            "use_spectator": False,
            "video_dir": None,
            "restrict_players": True,
            "ssh_args": [None for _ in range(num_players)],
        },
        "players": [
            {
                "is_human": True,
                "give_items": [
                    {
                        "id": "diamond_pickaxe",
                        "count": 1,
                        "enchantments": [
                            # Gives silk touch enchantment, level defaults to max.
                            {
                                "id": 33,
                            },
                            {
                                "id": 34,  # Gives unbreaking enchantment.
                                "level": 3,  # Manually set the level.
                            },
                        ],
                    }
                ],
            }
            for _ in range(num_players)
        ],
        "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
    }


@ex.automain
def main(
    launch_minecraft: bool,
    data_path: str,
    num_players: int,
    mbag_config: MbagConfigDict,
):
    os.makedirs(data_path, exist_ok=True)
    result_folder = os.path.join(
        data_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.mkdir(result_folder)

    minecraft_process: Optional[Popen] = None
    if launch_minecraft:
        (minecraft_process,) = minecraft.launch()

    evaluator = MbagEvaluator(
        mbag_config,
        [(HumanAgent, {}) for _ in range(num_players)],
        return_on_exception=True,
    )

    episode_info = evaluator.rollout()
    with open(os.path.join(result_folder, "result.json"), "w") as result_file:
        json.dump(
            episode_info.to_json_with_history(), result_file, cls=SafeFallbackEncoder
        )

    logger.info(f"saved file in {result_folder}")

    if minecraft_process is not None:
        minecraft_process.terminate()
