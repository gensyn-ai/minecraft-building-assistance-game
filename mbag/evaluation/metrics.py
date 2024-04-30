from typing import List, TypedDict, cast

import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MBAG_ACTION_BREAK_PALETTE_NAME, MbagAction

from .episode import MbagEpisode


class MbagPlayerMetrics(TypedDict, total=False):
    num_noop: int
    num_unintentional_noop: int
    num_break_block: int
    num_break_palette: int
    num_place_block: int
    num_give_block: int
    num_move_pos_x: int
    num_move_neg_x: int
    num_move_pos_y: int
    num_move_neg_y: int
    num_move_pos_z: int
    num_move_neg_z: int

    num_correct_break_block: int
    num_correct_place_block: int
    break_block_accuracy: float
    place_block_accuracy: float

    own_reward: float
    own_reward_prop: float
    goal_dependent_reward: float
    goal_independent_reward: float


class MbagEpisodeMetrics(TypedDict):
    player_metrics: List[MbagPlayerMetrics]
    goal_similarity: float
    goal_distance: float


def calculate_metrics(episode: MbagEpisode) -> MbagEpisodeMetrics:
    width, height, depth = episode.env_config["world_size"]

    goal_similarity = episode.last_infos[0]["goal_similarity"]
    goal_distance = width * height * depth - goal_similarity

    players_metrics: List[MbagPlayerMetrics] = []
    for player_index in range(episode.env_config["num_players"]):
        player_metrics: MbagPlayerMetrics = {}

        for valid_action_type in MbagActionDistribution.get_valid_action_types(
            episode.env_config
        ):
            action_type_name = MbagAction.ACTION_TYPE_NAMES[valid_action_type]
            player_metrics[f"num_{action_type_name.lower()}"] = 0  # type: ignore[literal-required]
            if valid_action_type in [MbagAction.BREAK_BLOCK, MbagAction.PLACE_BLOCK]:
                player_metrics[f"num_correct_{action_type_name.lower()}"] = 0  # type: ignore[literal-required]
        player_metrics["num_unintentional_noop"] = 0
        if not episode.env_config["abilities"]["inf_blocks"]:
            player_metrics["num_break_palette"] = 0
        player_metrics["own_reward"] = 0
        player_metrics["goal_dependent_reward"] = 0
        player_metrics["goal_independent_reward"] = 0

        for t in range(episode.length):
            info_dict = episode.info_history[t][player_index]

            player_metrics["own_reward"] += info_dict["own_reward"]
            player_metrics["goal_dependent_reward"] += info_dict[
                "goal_dependent_reward"
            ]
            player_metrics["goal_independent_reward"] += info_dict[
                "goal_independent_reward"
            ]

            action = info_dict["action"]
            if action.action_type == MbagAction.BREAK_BLOCK and action.is_palette(
                episode.env_config["abilities"]["inf_blocks"]
            ):
                action_type_name = MBAG_ACTION_BREAK_PALETTE_NAME
            else:
                action_type_name = MbagAction.ACTION_TYPE_NAMES[action.action_type]
            player_metrics[f"num_{action_type_name.lower()}"] += 1  # type: ignore[literal-required]

            if (
                info_dict["attempted_action"].action_type != MbagAction.NOOP
                and info_dict["action"].action_type == MbagAction.NOOP
            ):
                player_metrics["num_unintentional_noop"] += 1

            if info_dict["action_correct"]:
                player_metrics[f"num_correct_{action_type_name.lower()}"] += 1  # type: ignore[literal-required]

        last_info_dict = episode.last_infos[player_index]
        player_metrics["own_reward_prop"] = last_info_dict["own_reward_prop"]

        action_type_names = [
            MbagAction.ACTION_TYPE_NAMES[action_type]
            for action_type in [
                MbagAction.BREAK_BLOCK,
                MbagAction.PLACE_BLOCK,
            ]
        ]
        for action_type_name in action_type_names:
            num_correct = cast(
                int, player_metrics.get(f"num_correct_{action_type_name.lower()}", 0)
            )
            total: int = cast(
                int, player_metrics.get(f"num_{action_type_name.lower()}", 0)
            )
            percent_correct = num_correct / total if total != 0 else np.nan
            player_metrics[f"{action_type_name.lower()}_accuracy"] = percent_correct  # type: ignore[literal-required]

        players_metrics.append(player_metrics)

    return {
        "goal_similarity": episode.last_infos[0]["goal_similarity"],
        "goal_distance": goal_distance,
        "player_metrics": players_metrics,
    }


def calculate_mean_metrics(
    episodes_metrics: List[MbagEpisodeMetrics],
) -> MbagEpisodeMetrics:
    num_players = len(episodes_metrics[0]["player_metrics"])
    mean_player_metrics: List[MbagPlayerMetrics] = [{} for _ in range(num_players)]
    for player_index in range(num_players):
        mean_player_metrics[player_index] = {}
        for metric_name in episodes_metrics[0]["player_metrics"][player_index]:
            metric_values = [
                episode_metrics["player_metrics"][player_index][metric_name]  # type: ignore[literal-required]
                for episode_metrics in episodes_metrics
            ]
            mean_player_metrics[player_index][metric_name] = np.nanmean(metric_values)  # type: ignore[literal-required]
    return {
        "goal_similarity": np.mean(
            [episode_metrics["goal_similarity"] for episode_metrics in episodes_metrics]
        ),
        "goal_distance": np.mean(
            [episode_metrics["goal_distance"] for episode_metrics in episodes_metrics]
        ),
        "player_metrics": mean_player_metrics,
    }
