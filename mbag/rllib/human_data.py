import copy
import random
import zipfile
from datetime import datetime, timedelta
from typing import Optional, Sequence, Union

import numpy as np
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.policy.sample_batch import SampleBatch

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.compatibility_utils import OldHumanDataUnpickler
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import DEFAULT_CONFIG, MbagConfigDict
from mbag.environment.types import CURRENT_PLAYER, GOAL_BLOCKS, PLAYER_LOCATIONS
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


def convert_episode_info_to_sample_batch(  # noqa: C901
    episode_info: EpisodeInfo,
    *,
    player_index: int,
    # Players to include in the inventory observations.
    inventory_player_indices: Optional[Sequence[int]] = None,
    participant_id: int = -1,
    episode_dir: str = "",
    mbag_config: Optional[MbagConfigDict] = None,
    offset_rewards=False,
    place_wrong_reward: float = 0,
    include_noops=True,
    include_noops_for_other_player_actions=True,
    flat_actions=False,
    flat_observations=False,
    action_delay=DEFAULT_CONFIG["malmo"]["action_delay"],
) -> SampleBatch:
    if mbag_config is None:
        mbag_config = episode_info.env_config

    if inventory_player_indices is None:
        inventory_player_indices = range(mbag_config["num_players"])

    if mbag_config["rewards"]["own_reward_prop"] != 0:
        raise ValueError("This function only supports own_reward_prop=0.")

    sample_batch_builder = SampleBatchBuilder()
    episode_id = random.randrange(int(1e18))
    t = 0
    # Keep track of any rewards received during intermediate NOOPs.
    intermediate_rewards: float = 0
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
        if include_noops_for_other_player_actions:
            not_noop = any(action.action_type != MbagAction.NOOP for action in actions)
        else:
            not_noop = action.action_type != MbagAction.NOOP
        world_obs = obs[0]

        for other_player_index in range(mbag_config["num_players"]):
            other_info = episode_info.info_history[i][other_player_index]
            other_action = other_info["action"]
            if (
                other_action.action_type == MbagAction.PLACE_BLOCK
                and not other_info["action_correct"]
                and world_obs[(GOAL_BLOCKS,) + other_action.block_location]
                != MinecraftBlocks.AIR
            ):
                reward += place_wrong_reward
            if (
                other_action.action_type == MbagAction.BREAK_BLOCK
                and other_info["action_correct"]
                and world_obs[(GOAL_BLOCKS,) + other_action.block_location]
                != MinecraftBlocks.AIR
            ):
                reward -= place_wrong_reward

        intermediate_rewards += reward

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
                for other_player_index in inventory_player_indices:
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
                            SampleBatch.REWARDS: intermediate_rewards,
                            SampleBatch.DONES: False,
                            SampleBatch.INFOS: {},
                            PARTICIPANT_ID: participant_id,
                            EPISODE_DIR: episode_dir,
                        }
                    )
                    intermediate_rewards = 0
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
                    SampleBatch.REWARDS: intermediate_rewards,
                    SampleBatch.DONES: False,
                    SampleBatch.INFOS: info,
                    PARTICIPANT_ID: participant_id,
                    EPISODE_DIR: episode_dir,
                }
            )
            intermediate_rewards = 0
            t += 1
    return sample_batch_builder.build_and_reset()


def repair_missing_player_locations(
    episode_info: EpisodeInfo,
    *,
    mbag_config: Optional[MbagConfigDict] = None,
) -> EpisodeInfo:
    """
    In some of the human data, the observations seem to be missing the current player's
    location. This function tries to repair the missing locations by using the
    previous locations combined with the actions taken by the player.
    """

    repaired_episode_info = copy.deepcopy(episode_info)

    if mbag_config is None:
        mbag_config = episode_info.env_config
    width, height, depth = mbag_config["world_size"]
    num_players = mbag_config.get("num_players", len(episode_info.obs_history[0]))

    for player_index in range(num_players):
        prev_world_obs, _, _ = repaired_episode_info.obs_history[0][player_index]
        if not np.any(prev_world_obs[PLAYER_LOCATIONS] == CURRENT_PLAYER):
            raise ValueError(
                f"Player {player_index} location is missing in the first observation"
            )

        for t in range(1, episode_info.length):
            world_obs, _, _ = repaired_episode_info.obs_history[t][player_index]
            prev_action = repaired_episode_info.info_history[t - 1][player_index][
                "action"
            ]

            if np.any(world_obs[PLAYER_LOCATIONS] == CURRENT_PLAYER):
                pass  # The player location is already present in the observation.
            else:
                prev_xs, prev_ys, prev_zs = np.where(
                    prev_world_obs[PLAYER_LOCATIONS] == CURRENT_PLAYER
                )
                (prev_x,) = set(prev_xs)
                (prev_z,) = set(prev_zs)
                prev_y = np.min(prev_ys)

                x, y, z = prev_x, prev_y, prev_z
                if prev_action.action_type == MbagAction.MOVE_POS_X:
                    x += 1
                elif prev_action.action_type == MbagAction.MOVE_NEG_X:
                    x -= 1
                elif prev_action.action_type == MbagAction.MOVE_POS_Y:
                    y += 1
                elif prev_action.action_type == MbagAction.MOVE_NEG_Y:
                    y -= 1
                elif prev_action.action_type == MbagAction.MOVE_POS_Z:
                    z += 1
                elif prev_action.action_type == MbagAction.MOVE_NEG_Z:
                    z -= 1

                assert 0 <= x < width and 0 <= y < height and 0 <= z < depth
                world_obs[PLAYER_LOCATIONS, x, y, z] = CURRENT_PLAYER
                if y + 1 < height:
                    world_obs[PLAYER_LOCATIONS, x, y + 1, z] = CURRENT_PLAYER

            prev_world_obs = world_obs

    return repaired_episode_info
