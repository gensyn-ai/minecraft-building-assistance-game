import subprocess
import sys, getopt
import logging

from mbag.agents.heuristic_agents import NoopAgent
from mbag.agents.human_agent import HumanAgent
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.evaluation.evaluator import MbagEvaluator

logger = logging.getLogger(__name__)

READY_STATE = "CLIENT enter state: WAITING_FOR_MOD_READY"


def main(argv):
    try:
        ops, args = getopt.getopt(argv, "m:d:a", ["minecraft=", "data=", "assistant="])
    except getopt.GetoptError:
        print(
            "startHumanTrial.py -m <minecraft file> -d <data file> -a <True for assistant, False for builder>"
        )
        sys.exit(2)

    minecraftPath = (
        "/Users/timg/Documents/GitHub/MalmoPlatform/Minecraft/launchClient.sh"
    )
    dataPath = "./human_data"
    assistant = False
    trialBegan = False

    for opt, arg in ops:
        if opt in ("-m", "--minecraft"):
            minecraftPath = arg
        elif opt in ("-d", "--data"):
            dataPath = arg
        elif opt in ("-d", "--data"):
            assistant = arg

    print(minecraftPath)
    print(dataPath)

    process = subprocess.Popen(
        minecraftPath, stdout=subprocess.PIPE, universal_newlines=True
    )

    while True:
        output = process.stdout.readline()

        data = output.strip()
        print("Minecraft Client Log:", data)
        if READY_STATE in data and not trialBegan:
            launchHumanTrial()
            trialBegan = True

        return_code = process.poll()
        if return_code is not None:
            print("RETURN CODE", return_code)
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                print(output.strip())
            break


def launchHumanTrial():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 50,
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
        },
        [
            (HumanAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    print(episode_info)


if __name__ == "__main__":
    main(sys.argv[1:])
