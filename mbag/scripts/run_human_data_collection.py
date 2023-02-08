import subprocess
import logging
import json
import os

from datetime import datetime
from sacred import Experiment
from mbag.agents.heuristic_agents import NoopAgent
from mbag.agents.human_agent import HumanAgent
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.evaluation.evaluator import MbagEvaluator
from ray.tune.utils.util import SafeFallbackEncoder

logger = logging.getLogger(__name__)

READY_STATE = "CLIENT enter state: WAITING_FOR_MOD_READY"

ex = Experiment()


@ex.config
def make_human_action_config():
    minecraftPath = (
        "/Users/timg/Documents/GitHub/MalmoPlatform/Minecraft/launchClient.sh"
    )
    data_path = "data/logs/human_play"
    human = True
    horizon = 50


@ex.capture
def launchHumanTrial(result_folder, horizon, human):
    evaluator = MbagEvaluator(
        {
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
                    "is_human": human,
                }
            ],
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (HumanAgent, {}),
        ],
    )

    episode_info = evaluator.rollout()
    with open(os.path.join(result_folder, "result.json"), "w") as result_file:
        json.dump(episode_info.toJSON(), result_file, cls=SafeFallbackEncoder)

    logger.info("Saved file in ", result_folder)


@ex.automain
def main(minecraftPath, data_path, human):
    trialBegan = False

    result_folder = os.path.join(data_path, str(datetime.now()))
    os.mkdir(result_folder)

    # with open(os.path.join(result_folder, "config.json"), "w+") as result_file:
    #     json.dump([1, 2, 3, 4], result_file)

    process = subprocess.Popen(
        minecraftPath, stdout=subprocess.PIPE, universal_newlines=True
    )

    while True:
        output = process.stdout.readline()

        data = output.strip()
        logger.info("Minecraft Client Log:", data)
        if READY_STATE in data and not trialBegan:
            launchHumanTrial(result_folder)
            trialBegan = True

        return_code = process.poll()
        if return_code is not None:
            print("RETURN CODE", return_code)
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                print(output.strip())
            break
