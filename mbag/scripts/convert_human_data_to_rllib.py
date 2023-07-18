import logging
import os
import pickle
import random
import zipfile
from typing import List, Union

from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from sacred import Experiment

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import MbagAction, MbagActionTuple
from mbag.evaluation.evaluator import EpisodeInfo

ex = Experiment()


@ex.config
def sacred_config():
    data_dir = ""

    mbag_config: MbagConfigDict = {  # noqa: F841
        "world_size": (11, 10, 10),
        "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
    }
    include_noops = False  # noqa: F841
    flat_actions = True  # noqa: F841
    player_indices = [0]  # noqa: F841

    experiment_name = "rllib"
    experiment_name += "_with_noops" if include_noops else "_no_noops"
    experiment_name += "_flat_actions" if flat_actions else "_tuple_actions"
    experiment_name += f"_player_{' '.join(map(str, player_indices))}"
    out_dir = os.path.join(data_dir, experiment_name)  # noqa: F841


@ex.automain
def main(
    data_dir: str,
    out_dir: str,
    mbag_config: MbagConfigDict,
    include_noops: bool,
    flat_actions: bool,
    player_indices: List[int],
    _log: logging.Logger,
):
    episode_info: EpisodeInfo
    result_fname = os.path.join(data_dir, "episode.zip")
    if os.path.exists(result_fname):
        _log.info(f"reading {result_fname}...")
        with zipfile.ZipFile(result_fname, "r") as zip_file:
            with zip_file.open("episode.pkl", "r") as result_file:
                episode_info = pickle.load(result_file)
    else:
        result_fname = os.path.join(data_dir, "episode.pkl")

        _log.info(f"reading {result_fname}...")
        with open(result_fname, "rb") as result_file:
            episode_info = pickle.load(result_file)

    if hasattr(episode_info, "env_config"):
        _log.info("using env config from EpisodeInfo")
        mbag_config = episode_info.env_config

    sample_batch_builder = SampleBatchBuilder()
    json_writer = JsonWriter(out_dir)
    for player_index in player_indices:
        _log.info(f"converting to RLlib format for player {player_index}...")
        episode_id = random.randrange(int(1e18))
        t = 0
        for i in range(episode_info.length):
            obs = episode_info.obs_history[i][player_index]
            info = episode_info.info_history[i][player_index]
            reward = episode_info.reward_history[i]
            action = info["action"]
            if include_noops or action.action_type != MbagAction.NOOP:
                action_id: Union[int, MbagActionTuple]
                if flat_actions:
                    action_id = MbagActionDistribution.get_flat_action(
                        mbag_config, action.to_tuple()
                    )
                else:
                    action_id = action.to_tuple()
                sample_batch_builder.add_values(
                    t=t,
                    eps_id=episode_id,
                    agent_index=player_index,
                    obs=obs,
                    actions=action_id,
                    action_prob=1.0,
                    action_logp=0.0,
                    rewards=reward,
                    dones=False,
                    infos=info,
                )
                t += 1
        _log.info("saving trajectory...")
        sample_batch = sample_batch_builder.build_and_reset()
        json_writer.write(sample_batch)

    return {"mbag_config": mbag_config, "out_dir": out_dir}
