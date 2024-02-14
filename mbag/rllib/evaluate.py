import copy
import json
import os
import pickle
import random
from datetime import datetime
from logging import Logger
from typing import List, Optional, Type

import numpy as np
import ray
import torch
import tqdm
from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.typing import PolicyID
from sacred import SETTINGS, Experiment

from mbag.agents.human_agent import HumanAgent
from mbag.environment.mbag_env import MbagConfigDict
from mbag.evaluation.evaluator import EpisodeInfo, MbagAgentConfig, MbagEvaluator

from .agents import RllibAlphaZeroAgent, RllibMbagAgent
from .os_utils import available_cpu_count
from .training_utils import load_trainer

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS.CONFIG


ex = Experiment("evaluate")


@ex.config
def sacred_config():
    runs = ["PPO"]  # noqa: F841
    checkpoints = [""]  # noqa: F841
    policy_ids = [""]  # noqa: F841
    episodes = 100  # noqa: F841
    experiment_name = ""  # noqa: F841
    seed = 0  # noqa: F841
    record_video = False  # noqa: F841
    use_malmo = record_video  # noqa: F841

    config_updates = {}  # noqa: F841


@ex.automain
def main(
    runs: List[str],
    checkpoints: List[str],
    policy_ids: List[PolicyID],
    episodes: int,
    experiment_name: str,
    config_updates: dict,
    seed: int,
    record_video: bool,
    use_malmo: bool,
    _log: Logger,
):
    ray.init(
        num_cpus=available_cpu_count(),
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    config_updates["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config_updates.setdefault("num_workers", 0)
    config_updates.setdefault("num_envs_per_worker", 1)
    config_updates.setdefault("evaluation_num_workers", 0)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not experiment_name.endswith("_") and experiment_name != "":
        experiment_name += "_"
    out_path = os.path.join(
        os.path.dirname(checkpoints[0]), f"evaluate_{experiment_name}{time_str}"
    )

    env_config: Optional[MbagConfigDict] = None
    agent_configs: List[MbagAgentConfig] = []
    trainers: List[Algorithm] = []
    for run, checkpoint, policy_id in zip(runs, checkpoints, policy_ids):
        if run == "HumanAgent":
            agent_configs.append((HumanAgent, {}))
        else:
            _log.info(f"loading policy {policy_id} from {checkpoint}...")
            trainer = load_trainer(checkpoint, run, config_updates)
            policy = trainer.get_policy(policy_id)
            policy.config["player_index"] = 0  # TODO: remove

            mbag_agent_class: Type[RllibMbagAgent] = RllibMbagAgent
            if "AlphaZero" in run:
                mbag_agent_class = RllibAlphaZeroAgent

            agent_configs.append((mbag_agent_class, {"policy": policy}))
            trainers.append(trainer)

            if env_config is None:
                env_config = copy.deepcopy(trainer.config["env_config"])

    assert env_config is not None
    env_config["num_players"] = len(agent_configs)
    while len(env_config["players"]) < env_config["num_players"]:
        env_config["players"].append(copy.deepcopy(env_config["players"][0]))
        player_index = len(env_config["players"]) - 1
        env_config["players"][-1]["player_name"] = f"player_{player_index}"
    for player_index, (agent_class, agent_config) in enumerate(agent_configs):
        if agent_class is HumanAgent:
            env_config["players"][player_index]["is_human"] = True

    env_config.setdefault("malmo", {})
    env_config["malmo"]["use_malmo"] = use_malmo
    if record_video:
        env_config["malmo"]["video_dir"] = out_path

    _log.info(f"evaluating for {episodes} episodes...")
    evaluator = MbagEvaluator(
        env_config,
        agent_configs,
    )

    episode_infos: List[EpisodeInfo] = []
    for _ in tqdm.trange(episodes):
        episode_infos.append(evaluator.rollout())

    out_pickle_fname = out_path + ".pickle"
    _log.info(f"saving full results to {out_pickle_fname}")
    with open(out_pickle_fname, "wb") as out_pickle:
        pickle.dump(episode_infos, out_pickle)

    out_json_fname = out_path + ".json"
    _log.info(f"saving JSON results to {out_json_fname}")
    with open(out_json_fname, "w") as out_json:
        json.dump([episode_info.to_json() for episode_info in episode_infos], out_json)

    for trainer in trainers:
        trainer.stop()
