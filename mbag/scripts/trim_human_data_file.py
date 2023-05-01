import logging
import os
import pickle
from datetime import datetime
from subprocess import Popen
from typing import List, Optional
import numpy as np

from sacred import Experiment

from mbag.agents.human_agent import HumanAgent
from mbag.environment.goals import TransformedGoalGenerator
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import MbagAction, MbagInfoDict, MbagObs
from mbag.evaluation.evaluator import EpisodeInfo, MbagEvaluator

logger = logging.getLogger(__name__)

ex = Experiment()


@ex.config
def make_trim_data_config():
    result_folder = "/Users/timg/Documents/GitHub/minecraft-building-assistance-game/data/human_data/2023-04-11_12-42-43"


def filter(
    reward: float,
    info_dicts: List[MbagInfoDict],
    obs: List[MbagObs],
    next_obs: List[MbagObs],
) -> bool:
    """
    Returns whether the timestamp should be trimmed.

    Trim timestamps with zero reward and zero human actions
    """

    if reward != 0:
        return True

    for info_dict in info_dicts:
        if info_dict["attempted_action"].action_type != MbagAction.NOOP:
            return True

        if len(info_dict["human_actions"]) > 0:
            return True

    for i in range(len(obs)):
        current_player_obs = obs[i]
        next_player_obs = next_obs[i]
        if not (
            np.array_equal(current_player_obs[0], next_player_obs[0])
            and np.array_equal(current_player_obs[1], next_player_obs[1])
        ):
            return True

    return False


@ex.automain
def main(
    result_folder: str,
):
    result_path = os.path.join(result_folder, "result.pb")
    compressed_path = os.path.join(result_folder, "result_compressed.pb")

    with open(result_path, "rb") as result_file:
        episode_info = pickle.load(result_file)

    logger.info(f"Opening {result_path}")
    logger.info(f"Old file size: {os.stat(result_path).st_size / (1024 * 1024)} MB")

    trimmed_rewards: List[float] = []
    trimmed_obs: List[List[MbagObs]] = []
    trimmed_infos: List[List[MbagObs]] = []

    for i in range(len(episode_info.info_history)):
        if filter(
            episode_info.reward_history[i],
            episode_info.info_history[i],
            episode_info.obs_history[i],
            episode_info.obs_history[i + 1],
        ):
            trimmed_rewards.append(episode_info.reward_history[i])
            trimmed_obs.append(episode_info.obs_history[i])
            trimmed_infos.append(episode_info.info_history[i])

    episode_info.reward_history = trimmed_rewards
    episode_info.info_history = trimmed_infos
    episode_info.obs_history = trimmed_obs

    with open(compressed_path, "wb") as result_file:
        pickle.dump(episode_info, result_file)

    logger.info(f"New file size: {os.stat(compressed_path).st_size / (1024 * 1024)} MB")
    logger.info(f"Saved compressed file in {compressed_path}")
