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
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.environment.mbag_env import MbagConfigDict
from mbag.evaluation.evaluator import MbagEvaluator

logger = logging.getLogger(__name__)

READY_STATE = "CLIENT enter state: WAITING_FOR_MOD_READY"

ex = Experiment()


@ex.config
def make_human_action_config():
    launch_minecraft = False  # noqa: F841
    data_path = "data/human_data"  # noqa: F841
    horizon = 50

    mbag_config: MbagConfigDict = {  # noqa: F841
        "world_size": (5, 6, 5),
        "num_players": 1,
        "horizon": horizon,
        "goal_generator": BasicGoalGenerator,
        "goal_generator_config": {"pallette": True},
        "malmo": {
            "use_malmo": True,
            "use_spectator": False,
            "video_dir": None,
        },
        "players": [
            {
                "is_human": True,
            }
        ],
        "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
    }


@ex.automain
def main(
    launch_minecraft: bool, data_path: str, human: bool, mbag_config: MbagConfigDict
):
    result_folder = os.path.join(data_path, str(datetime.now()))
    os.mkdir(result_folder)

    minecraft_process: Optional[Popen] = None
    if launch_minecraft:
        (minecraft_process,) = minecraft.launch()

    evaluator = MbagEvaluator(
        mbag_config,
        [
            (HumanAgent, {}),
        ],
        return_on_exception=True,
    )

    episode_info = evaluator.rollout()
    with open(os.path.join(result_folder, "result.json"), "w") as result_file:
        json.dump(episode_info.toJSON(), result_file, cls=SafeFallbackEncoder)

    logger.info("Saved file in ", result_folder)

    if minecraft_process is not None:
        minecraft_process.terminate()
