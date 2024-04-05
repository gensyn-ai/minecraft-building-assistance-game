import glob
import logging
import os
import random
import zipfile
from typing import List, Union

import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import SampleBatch
from sacred import Experiment

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.compatibility_utils import (
    OldHumanDataUnpickler,
    convert_old_rewards_config_to_new,
)
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.mbag_env import MbagConfigDict
from mbag.evaluation.evaluator import EpisodeInfo

from .human_data import EPISODE_DIR, PARTICIPANT_ID

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
    flat_observations = True  # noqa: F841
    offset_rewards = False  # noqa: F841
    player_indices = [0]  # noqa: F841

    experiment_name = "rllib"
    experiment_name += "_with_noops" if include_noops else "_no_noops"
    experiment_name += "_flat_actions" if flat_actions else "_tuple_actions"
    experiment_name += "_flat_observations" if flat_observations else ""
    experiment_name += f"_player_{'_'.join(map(str, player_indices))}"
    out_dir = os.path.join(data_dir, experiment_name)  # noqa: F841


@ex.automain
def main(  # noqa: C901
    data_dir: str,
    out_dir: str,
    mbag_config: MbagConfigDict,
    include_noops: bool,
    flat_actions: bool,
    flat_observations: bool,
    offset_rewards: bool,
    player_indices: List[int],
    _log: logging.Logger,
):
    episode_info: EpisodeInfo

    episode_fnames = glob.glob(
        os.path.join(data_dir, "**", "episode.pkl"), recursive=True
    ) + glob.glob(os.path.join(data_dir, "**", "episode.zip"), recursive=True)
    if not episode_fnames:
        raise FileNotFoundError(f"No episode files found in {data_dir}.")

    if os.path.exists(out_dir):
        raise FileExistsError(f"Output directory {out_dir} already exists.")

    sample_batch_builder = SampleBatchBuilder()
    json_writer = JsonWriter(out_dir)

    for episode_fname in sorted(episode_fnames):
        try:
            if episode_fname.endswith(".zip"):
                _log.info(f"reading {episode_fname}...")
                with zipfile.ZipFile(episode_fname, "r") as zip_file:
                    with zip_file.open("episode.pkl", "r") as episode_file:
                        episode_info = OldHumanDataUnpickler(episode_file).load()
            else:
                _log.info(f"reading {episode_fname}...")
                with open(episode_fname, "rb") as episode_file:
                    episode_info = OldHumanDataUnpickler(episode_file).load()
        except Exception:
            _log.exception(f"failed to read {episode_fname}")
            continue

        assert episode_fname[: len(data_dir)] == data_dir
        episode_dir = os.path.dirname(episode_fname)[len(data_dir) :].lstrip(
            os.path.sep
        )
        participant_id = -1
        for path_part in episode_dir.split(os.path.sep):
            if path_part.startswith("participant_"):
                participant_id = int(path_part[len("participant_") :])
                break

        if hasattr(episode_info, "env_config"):
            _log.info("using env config from EpisodeInfo")
            mbag_config = episode_info.env_config

        if "rewards" in mbag_config:
            mbag_config["rewards"] = convert_old_rewards_config_to_new(
                mbag_config["rewards"]
            )
        for player_config in mbag_config.get("players", []):
            if "rewards" in player_config:
                player_config["rewards"] = convert_old_rewards_config_to_new(
                    player_config["rewards"]
                )

        for player_index in player_indices:
            _log.info(f"converting to RLlib format for player {player_index}...")
            episode_id = random.randrange(int(1e18))
            t = 0
            for i in range(episode_info.length):
                obs = episode_info.obs_history[i][player_index]
                info = episode_info.info_history[i][player_index]
                reward = episode_info.reward_history[i]
                if offset_rewards:
                    reward = episode_info.reward_history[i + 1]
                assert (
                    reward
                    == info["goal_dependent_reward"] + info["goal_independent_reward"]
                )
                action = info["action"]
                if include_noops or action.action_type != MbagAction.NOOP:
                    action_id: Union[int, MbagActionTuple]
                    if flat_actions:
                        action_id = MbagActionDistribution.get_flat_action(
                            mbag_config, action.to_tuple()
                        )
                    else:
                        action_id = action.to_tuple()
                    obs = obs[0], obs[1], np.array(t)
                    if flat_observations:
                        obs = np.concatenate([obs_piece.flat for obs_piece in obs])
                    sample_batch_builder.add_values(
                        **{
                            SampleBatch.T: t,
                            SampleBatch.EPS_ID: episode_id,
                            SampleBatch.AGENT_INDEX: player_index,
                            SampleBatch.OBS: obs,
                            SampleBatch.ACTIONS: action_id,
                            SampleBatch.ACTION_PROB: 1.0,
                            SampleBatch.ACTION_LOGP: 0.0,
                            SampleBatch.REWARDS: reward,
                            SampleBatch.DONES: False,
                            SampleBatch.INFOS: info,
                            PARTICIPANT_ID: participant_id,
                            EPISODE_DIR: episode_dir,
                        }
                    )
                    t += 1
            if t == 0:
                _log.info("skipping empty trajectory")
                continue
            _log.info("saving trajectory...")
            sample_batch = sample_batch_builder.build_and_reset()
            total_reward = sample_batch["rewards"].sum()
            _log.info(
                "episode info: participant ID=%d length=%d total reward=%.1f",
                participant_id,
                len(sample_batch),
                total_reward,
            )
            assert total_reward == episode_info.cumulative_reward
            json_writer.write(sample_batch)

    return {"mbag_config": mbag_config, "out_dir": out_dir}
