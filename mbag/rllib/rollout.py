import json
import os
from datetime import datetime
from typing import List, Optional

import gymnasium as gym
import ray
import torch
from ray.rllib.evaluate import RolloutSaver
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts  # type: ignore
from ray.rllib.utils.typing import PolicyID
from ray.tune.utils.util import SafeFallbackEncoder
from sacred import SETTINGS, Experiment
from .os_utils import available_cpu_count

from .training_utils import load_trainer

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


ex = Experiment("rollout")


@ex.config
def sacred_config():
    run = "MbagPPO"  # noqa: F841
    checkpoint = ""  # noqa: F841
    episodes = 100  # noqa: F841
    experiment_name = ""  # noqa: F841
    policy_ids: Optional[List[str]] = None  # noqa: F841
    player_names = policy_ids  # noqa: F841
    seed = 0

    experiment_tag = None
    if experiment_tag is not None:
        experiment_name += experiment_tag

    num_workers = 4
    output_max_file_size = 64 * 1024 * 1024
    config_updates = {  # noqa: F841
        "seed": seed,
        "evaluation_num_workers": num_workers,
        "create_env_on_driver": True,
        "evaluation_num_episodes": episodes,
        "output_max_file_size": output_max_file_size,
        "evaluation_config": {},
        "env_config": {"malmo": {}},
        "multiagent": {},
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "disable_env_checking": True,
        "evaluation_sample_timeout_s": 365 * 24 * 3600,
    }
    extra_config_updates = {}  # noqa: F841

    record_video = False  # noqa: F841

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(  # noqa: F841
        checkpoint, f"rollouts_{experiment_name}{time_str}"
    )


@ex.automain
def main(
    run: str,
    config_updates: dict,
    extra_config_updates: dict,
    checkpoint: str,
    experiment_name: str,
    policy_ids: Optional[List[str]],
    player_names: Optional[List[str]],
    record_video: bool,
    out_dir: str,
    _log,
):
    ray.init(
        num_cpus=available_cpu_count(),
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    config_updates = merge_dicts(
        config_updates,
        extra_config_updates,
    )

    if not experiment_name.endswith("_") and experiment_name != "":
        experiment_name += "_"
    _log.info(f"writing output to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    config_updates["output"] = out_dir

    if policy_ids is not None:

        def policy_mapping_fn(
            agent_id: str, episode=None, worker=None, policy_ids=policy_ids, **kwargs
        ):
            agent_index = int(agent_id[len("player_") :])
            return policy_ids[agent_index]

        config_updates["multiagent"]["policy_mapping_fn"] = policy_mapping_fn
        config_updates["env_config"]["num_players"] = len(policy_ids)

    if player_names is not None:
        assert "players" not in config_updates["env_config"]
        config_updates["env_config"]["players"] = [
            {"player_name": player_name} for player_name in player_names
        ]

    if record_video:
        config_updates.setdefault("evaluation_config", {})
        config_updates["evaluation_config"].setdefault("env_config", {})
        config_updates["evaluation_config"]["env_config"].setdefault("malmo", {})
        config_updates["evaluation_config"]["env_config"]["malmo"].update(
            {
                "use_malmo": True,
                "use_spectator": True,
                "video_dir": out_dir,
            }
        )
        config_updates["evaluation_num_workers"] = 1
        config_updates["create_env_on_local_worker"] = False

    trainer = load_trainer(checkpoint, run, config_updates)
    evaluation_workers: Optional[WorkerSet] = trainer.evaluation_workers

    if evaluation_workers is not None:
        # Remove the action_dist_inputs view requirement since it results in massive
        # (multi-gigabyte) JSON rollout files.
        def remove_action_dist_inputs_view_requirement(
            policy: Policy, policy_id: PolicyID
        ):
            if SampleBatch.ACTION_DIST_INPUTS in policy.view_requirements:
                del policy.view_requirements[SampleBatch.ACTION_DIST_INPUTS]

        evaluation_workers.foreach_policy(remove_action_dist_inputs_view_requirement)

    gym.logger.set_level(gym.logger.INFO)

    saver = RolloutSaver()

    saver.begin_rollout()
    eval_result = trainer.evaluate()["evaluation"]
    saver.end_rollout()
    trainer.stop()

    result_fname = os.path.join(out_dir, "result.json")
    _log.info(f"saving results to {result_fname}")
    with open(result_fname, "w") as result_file:
        json.dump(eval_result, result_file, cls=SafeFallbackEncoder)

    return eval_result
