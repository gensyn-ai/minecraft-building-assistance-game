from typing import Dict, Optional, cast
import numpy as np

from ray.rllib.algorithms.alpha_zero.alpha_zero import AlphaZeroDefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID

from mbag.environment.types import MbagAction, MbagInfoDict
from mbag.rllib.rllib_env import unwrap_mbag_env


class MbagCallbacks(AlphaZeroDefaultCallbacks):
    def on_episode_start(
        self,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        super().on_episode_start(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs,
        )

        env = base_env.get_sub_environments()[0]
        state = env.get_state()
        episode.user_data["state"] = state

        def update_env_global_timestep(env):
            if worker.global_vars is not None:
                unwrap_mbag_env(env).update_global_timestep(
                    worker.global_vars["timestep"]
                )

        worker.foreach_env(update_env_global_timestep)

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        assert policies is not None

        env = base_env.get_sub_environments()[0]
        state = env.get_state()
        episode.user_data["state"] = state

        for agent_id in episode.get_agents():
            policy_id = worker.policy_mapping_fn(agent_id, episode, worker)
            own_reward_key = f"{policy_id}/own_reward"
            if own_reward_key not in episode.custom_metrics:
                episode.custom_metrics[own_reward_key] = 0
            info_dict = cast(MbagInfoDict, episode.last_info_for(agent_id))
            episode.custom_metrics[own_reward_key] += info_dict["own_reward"]

            # Log what action the agent made
            action_type_name = MbagAction.ACTION_TYPE_NAMES[
                info_dict["action"].action_type
            ]
            action_key = f"{policy_id}/num_{action_type_name.lower()}"

            if action_key not in episode.custom_metrics:
                for action_name_ in MbagAction.ACTION_TYPE_NAMES.values():
                    episode.custom_metrics[
                        f"{policy_id}/num_{action_name_.lower()}"
                    ] = 0
            episode.custom_metrics[action_key] += 1

            if f"{policy_id}/num_correct_place_block" not in episode.custom_metrics:
                for name in ["place_block", "break_block"]:
                    episode.custom_metrics[f"{policy_id}/num_correct_{name}"] = 0

            if info_dict["action_correct"]:
                action_correct_key = (
                    f"{policy_id}/num_correct_{action_type_name.lower()}"
                )
                episode.custom_metrics[action_correct_key] += 1

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )

        info_dict = cast(MbagInfoDict, episode.last_info_for("player_0"))
        episode.custom_metrics["goal_similarity"] = info_dict["goal_similarity"]

        for agent_id in episode.get_agents():
            policy_id = worker.policy_mapping_fn(agent_id, episode, worker)
            info_dict = cast(MbagInfoDict, episode.last_info_for(agent_id))
            episode.custom_metrics[f"{policy_id}/own_reward_prop"] = info_dict[
                "own_reward_prop"
            ]

            for action_type in [MbagAction.BREAK_BLOCK, MbagAction.PLACE_BLOCK]:
                action_type_name = MbagAction.ACTION_TYPE_NAMES[action_type]
                num_correct = episode.custom_metrics.get(
                    f"{policy_id}/num_correct_{action_type_name.lower()}", 0
                )
                total = episode.custom_metrics.get(
                    f"{policy_id}/num_{action_type_name.lower()}", 0
                )
                percent_correct = num_correct / total if total != 0 else np.nan
                episode.custom_metrics[
                    f"{policy_id}/{action_type_name.lower()}_accuracy"
                ] = percent_correct
