import copy
import os
from datetime import datetime
from logging import Logger
from typing import Dict, List, Optional, Type, cast

import cloudpickle
import numpy as np
import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import (
    ModelConfigDict,
    MultiAgentPolicyConfigDict,
    PolicyState,
)
from ray.tune.registry import get_trainable_cls
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver

import mbag
from mbag.rllib.os_utils import available_cpu_count
from mbag.rllib.training_utils import build_logger_creator, load_trainer_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


ex = Experiment("create_mixture_model")


def find_shared_ancestor(paths):
    # Split each path into its components
    split_paths = [path.split(os.sep) for path in paths]

    # Find the minimum length path to limit the comparison
    min_length = min(len(p) for p in split_paths)

    shared_ancestor = []

    for i in range(min_length):
        # Get the i-th component of each path
        components = [p[i] for p in split_paths]

        # Check if all components are the same
        if all(component == components[0] for component in components):
            shared_ancestor.append(components[0])
        else:
            break

    return os.sep.join(shared_ancestor)


@ex.config
def sacred_config():
    run = "BC"  # noqa: F841
    checkpoints = []  # type: ignore

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_tag: Optional[str] = "mixture"

    out_dir = os.path.join(
        find_shared_ancestor(checkpoints),
        *([experiment_tag] if experiment_tag is not None else []),
        time_str,
    )
    observer = FileStorageObserver(out_dir)
    ex.observers.append(observer)


@ex.automain
def main(
    run: str,
    checkpoints: List[str],
    observer,
    _log: Logger,
):
    os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"] = "0"
    ray.init(
        num_cpus=available_cpu_count(),
        ignore_reinit_error=True,
        include_dashboard=False,
    )
    mbag.logger.setLevel(_log.getEffectiveLevel())

    config: AlgorithmConfig
    model_configs: List[ModelConfigDict] = []
    policy_states: List[PolicyState] = []

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        _log.info(f"Loading config and weights from {checkpoint}...")
        checkpoint_config: AlgorithmConfig = load_trainer_config(checkpoint)
        checkpoint_policies_config = checkpoint_config["multiagent"]["policies"]
        ((policy_id, policy_spec),) = checkpoint_policies_config.items()
        if not isinstance(policy_spec, PolicySpec):
            policy_spec = PolicySpec(*policy_spec)
        model_config = policy_spec.config["model"]
        model_configs.append(model_config)

        with open(
            os.path.join(checkpoint, "policies", policy_id, "policy_state.pkl"), "rb"
        ) as policy_state_file:
            policy_states.append(cloudpickle.load(policy_state_file))

        if checkpoint_index == 0:
            config = checkpoint_config.copy(copy_frozen=False)
            new_policy_spec = copy.deepcopy(policy_spec)
            new_policy_spec.config["model"]["custom_model"] = "mbag_mixture"
            policies_config: MultiAgentPolicyConfigDict = {policy_id: new_policy_spec}

    new_policy_spec.config["model"]["custom_model_config"] = {
        "model_configs": model_configs,
    }

    _log.info("Building new trainer...")
    if isinstance(run, str):
        algorithm_class = cast(Type[Algorithm], get_trainable_cls(run))
    else:
        algorithm_class = run
    config.policies = policies_config
    config.num_rollout_workers = 0
    config.evaluation_num_workers = 0
    trainer = algorithm_class(
        config=config,
        logger_creator=build_logger_creator(observer.dir),
    )

    _log.info("Loading weights into trainer...")
    ((policy_id, _),) = policies_config.items()
    policy_weights = {}
    for component_index, component_policy_state in enumerate(policy_states):
        component_policy_weights = cast(
            Dict[str, np.ndarray], component_policy_state["weights"]
        )
        for weight_key, weight in component_policy_weights.items():
            policy_weights[f"components.{component_index}.{weight_key}"] = weight
    trainer.set_weights(
        {
            policy_id: policy_weights,
        }
    )

    _log.info("Saving checkpoint...")
    checkpoint = trainer.save()
    _log.info(f"Saved checkpoint to {checkpoint}")

    trainer.stop()

    return {
        "final_checkpoint": checkpoint,
    }
