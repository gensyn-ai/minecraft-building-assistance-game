import glob
import os
from typing import Any, Callable, Dict, Type, Union, cast

import cloudpickle
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.utils.typing import PolicyID
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_trainable_cls

from mbag.compatibility_utils import convert_old_config_to_new
from mbag.environment.mbag_env import MbagEnv


def build_logger_creator(experiment_dir: str):
    def custom_logger_creator(config):
        """
        Creates a Unified logger that stores results in
        <log_dir>/<experiment_name>_<timestamp>
        """

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        return UnifiedLogger(config, experiment_dir)

    return custom_logger_creator


def load_trainer_config(checkpoint_path: str) -> AlgorithmConfig:
    checkpoint_info = get_checkpoint_info(checkpoint_path)
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    config: AlgorithmConfig = state["config"]
    return config


def load_trainer(
    checkpoint_path: str,
    run: Union[str, Type[Algorithm]],
    config_updates: dict = {},
    config_update_fn: Callable[
        [AlgorithmConfig], AlgorithmConfig
    ] = lambda config: config,
) -> Algorithm:
    config_updates.setdefault("num_workers", 0)

    checkpoint_info = get_checkpoint_info(checkpoint_path)
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    config: AlgorithmConfig = state["config"]
    config.update_from_dict(config_updates)

    config["env_config"] = convert_old_config_to_new(config["env_config"])

    env = MbagEnv(config["env_config"])
    observation_space = env.observation_space
    for policy_id in config.policies.keys():
        maybe_policy_spec = config.policies[policy_id]
        if isinstance(maybe_policy_spec, PolicySpec):
            policy_spec = maybe_policy_spec
        else:
            policy_spec = PolicySpec(*maybe_policy_spec)
        policy_spec.observation_space = observation_space
        policy_spec.config["num_gpus"] = config["num_gpus"]
        config.policies[policy_id] = policy_spec

    config = config_update_fn(config)

    # Create the Trainer from config.
    if isinstance(run, str):
        cls = cast(Type[Algorithm], get_trainable_cls(run))
    else:
        cls = run
    trainer: Algorithm = cls.from_state(state)
    # Load state from checkpoint.
    trainer.restore(checkpoint_path)

    return trainer


def load_policies_from_checkpoint(
    checkpoint_dir: str,
    trainer: Algorithm,
    policy_map: Callable[[PolicyID], PolicyID] = lambda policy_id: policy_id,
):
    """
    Load policy model weights from a checkpoint and copy them into the given
    trainer.
    """

    policy_dirs = glob.glob(os.path.join(checkpoint_dir, "policies", "*"))
    policy_states: Dict[str, Any] = {}
    for policy_dir in policy_dirs:
        policy_id = os.path.basename(policy_dir)
        with open(
            os.path.join(policy_dir, "policy_state.pkl"), "rb"
        ) as policy_state_file:
            policy_states[policy_id] = cloudpickle.load(policy_state_file)

    policy_weights = {
        policy_map(policy_id): policy_state["weights"]
        for policy_id, policy_state in policy_states.items()
    }

    def copy_policy_weights(policy: Policy, policy_id: PolicyID):
        if policy_id in policy_weights:
            policy.set_weights(policy_weights[policy_id])

    workers: WorkerSet = cast(Any, trainer).workers
    workers.foreach_policy(copy_policy_weights)
