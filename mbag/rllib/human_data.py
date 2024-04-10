import random
import zipfile
from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.policy.sample_batch import SampleBatch

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.compatibility_utils import OldHumanDataUnpickler
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.config import DEFAULT_CONFIG, MbagConfigDict
from mbag.evaluation.evaluator import EpisodeInfo

PARTICIPANT_ID = "participant_id"
EPISODE_DIR = "episode_dir"


def load_episode_info(episode_fname: str) -> EpisodeInfo:
    if episode_fname.endswith(".zip"):
        with zipfile.ZipFile(episode_fname, "r") as zip_file:
            with zip_file.open("episode.pkl", "r") as episode_file:
                episode_info = OldHumanDataUnpickler(episode_file).load()
    else:
        with open(episode_fname, "rb") as episode_file:
            episode_info = OldHumanDataUnpickler(episode_file).load()
    if not isinstance(episode_info, EpisodeInfo):
        raise ValueError(
            f"Invalid episode info in {episode_fname} ({type(episode_info)})"
        )
    return episode_info


def convert_episode_info_to_sample_batch(
    episode_info: EpisodeInfo,
    *,
    player_index: int,
    participant_id: int = -1,
    episode_dir: str = "",
    mbag_config: Optional[MbagConfigDict] = None,
    offset_rewards=False,
    include_noops=True,
    flat_actions=False,
    flat_observations=False,
    action_delay=DEFAULT_CONFIG["malmo"]["action_delay"],
) -> SampleBatch:
    if mbag_config is None:
        mbag_config = episode_info.env_config

    sample_batch_builder = SampleBatchBuilder()
    episode_id = random.randrange(int(1e18))
    t = 0
    prev_action_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    for i in range(episode_info.length):
        obs = episode_info.obs_history[i][player_index]
        info = episode_info.info_history[i][player_index]
        reward = episode_info.reward_history[i]
        if offset_rewards:
            reward = episode_info.reward_history[i + 1]
        assert reward == sum(
            info["goal_dependent_reward"] + info["goal_independent_reward"]
            for info in episode_info.info_history[i]
        )
        action = info["action"]
        actions = [info["action"] for info in episode_info.info_history[i]]
        not_noop = any(action.action_type != MbagAction.NOOP for action in actions)

        if info["malmo_observations"]:
            if prev_action_time is None:
                prev_action_time = info["malmo_observations"][0][0]
            current_time = info["malmo_observations"][-1][0]

        if (include_noops and action_delay == 0) or not_noop:
            action_id: Union[int, MbagActionTuple]
            if flat_actions:
                action_id = MbagActionDistribution.get_flat_action(
                    mbag_config, action.to_tuple()
                )
            else:
                action_id = action.to_tuple()
            world_obs = obs[0]
            inventory_obs = obs[1]
            if inventory_obs.ndim == 1:
                # Old observations, which may be present in old human data,
                # only had the block counts for the given player, not for
                # all players. We need to add the block counts for all players
                # in this case.
                inventory_obs_pieces = [inventory_obs]
                for other_player_index in range(mbag_config["num_players"]):
                    if other_player_index != player_index:
                        other_inventory = episode_info.obs_history[i][
                            other_player_index
                        ][1]
                        inventory_obs_pieces.append(other_inventory)
                inventory_obs = np.stack(inventory_obs_pieces, axis=0)
            obs = world_obs, inventory_obs, np.array(t)
            if flat_observations:
                obs = np.concatenate([obs_piece.flat for obs_piece in obs])

            # Add NOOPs based on action delay if necessary.
            if include_noops and action_delay > 0:
                assert prev_action_time is not None and current_time is not None
                while current_time > prev_action_time + timedelta(seconds=action_delay):
                    prev_action_time += timedelta(seconds=action_delay)
                    sample_batch_builder.add_values(
                        **{
                            SampleBatch.T: t,
                            SampleBatch.EPS_ID: episode_id,
                            SampleBatch.AGENT_INDEX: player_index,
                            SampleBatch.OBS: obs,
                            SampleBatch.ACTIONS: 0 if flat_actions else (0, 0, 0),
                            SampleBatch.ACTION_PROB: 1.0,
                            SampleBatch.ACTION_LOGP: 0.0,
                            SampleBatch.REWARDS: 0,
                            SampleBatch.DONES: False,
                            SampleBatch.INFOS: {},
                            PARTICIPANT_ID: participant_id,
                            EPISODE_DIR: episode_dir,
                        }
                    )
                    t += 1
                prev_action_time = current_time

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
    return sample_batch_builder.build_and_reset()
