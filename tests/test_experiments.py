"""
This files contains tests that run the entire experiment pipeline (with zero actual
training). It ensures that all configs match and that the pipeline runs without errors.
"""

import glob
import json
import logging
import os
from typing import Any, Dict

import pytest

import mbag.environment.goals
from mbag.scripts.train import ex as train_ex


def assert_config_matches(
    experiment_dir,
    reference_experiment_dir,
):
    with open(os.path.join(experiment_dir, "config.json")) as config_file:
        json_config = json.load(config_file)

    with open(os.path.join(reference_experiment_dir, "config.json")) as config_file:
        reference_json_config = json.load(config_file)

    for config in [json_config["config"], reference_json_config["config"]]:
        # We don't use rollout workers in the tests, so we remove them from the config.
        del config["num_rollout_workers"]
        del config["evaluation_num_workers"]

        # Don't report inconsistencies in the number of GPUs.
        del config["num_gpus"]
        del config["num_gpus_per_worker"]

        # Weird inconsistent JSON serialization.
        for policy_id, policy_spec in config["policies"].items():
            timestep_space = policy_spec["observation_space"]["spaces"]["py/tuple"][2]
            del timestep_space["bounded_above"]
            del timestep_space["dtype"]

    assert json_config["config"] == reference_json_config["config"]
    assert json_config["run"] == reference_json_config["run"]
    assert (
        json_config["load_policies_mapping"]
        == reference_json_config["load_policies_mapping"]
    )

    # General sanity checks to make sure the config is following best practices.
    env_config = json_config["config"]["env_config"]
    assert env_config["randomize_first_episode_length"] is True
    assert (
        json_config["evaluation_config"]["env_config"]["randomize_first_episode_length"]
        is False
    )
    assert env_config["horizon"] == 1500

    for policy_id, policy_spec in json_config["config"]["policies"].items():
        policy_config = policy_spec["config"]
        custom_model_config = policy_config.get("model", {}).get("custom_model_config")
        if custom_model_config:
            assert custom_model_config["line_of_sight_masking"]
            assert custom_model_config["mask_action_distribution"]
            assert custom_model_config["scale_obs"]
            assert custom_model_config["num_layers"] == 6
            assert custom_model_config["hidden_size"] == 64

    if json_config["run"] == "MbagAlphaZero":
        mcts_config = json_config["config"]["mcts_config"]
        assert mcts_config["use_bilevel_action_selection"]
        assert mcts_config["fix_bilevel_action_selection"]


@pytest.mark.timeout(600)
def test_experiments(tmp_path):
    # Supress huge number of logging messages about the goals being sampled.
    mbag.environment.goals.logger.setLevel(logging.WARNING)

    common_config_updates = {
        "num_training_iters": 0,
        "num_workers": 0,
        "evaluation_num_workers": 0,
    }

    ppo_human_result = train_ex.run(
        named_configs=["ppo_human"],
        config_updates={
            "experiment_dir": str(tmp_path / "ppo_human"),
            **common_config_updates,
        },
    ).result
    assert ppo_human_result is not None
    assert_config_matches(
        glob.glob(str(tmp_path / "ppo_human" / "[1-9]*"))[0],
        "data/testing/reference_experiments/ppo_human",
    )

    alphazero_human_result = train_ex.run(
        named_configs=["alphazero_human"],
        config_updates={
            "experiment_dir": str(tmp_path / "alphazero_human"),
            **common_config_updates,
        },
    ).result
    assert alphazero_human_result is not None
    assert_config_matches(
        glob.glob(str(tmp_path / "alphazero_human" / "[1-9]*"))[0],
        "data/testing/reference_experiments/alphazero_human",
    )

    bc_human_results: Dict[str, Any] = {}
    for bc_human_name, checkpoint_to_load_policies in [
        ("rand_init", None),
        ("ppo_init", ppo_human_result["final_checkpoint"]),
        ("alphazero_init", alphazero_human_result["final_checkpoint"]),
    ]:
        experiment_dir = tmp_path / f"bc_human_{bc_human_name}"
        bc_human_result = train_ex.run(
            named_configs=["bc_human"],
            config_updates={
                "experiment_dir": str(experiment_dir),
                "checkpoint_to_load_policies": checkpoint_to_load_policies,
                **common_config_updates,
            },
        ).result
        assert bc_human_result is not None
        assert_config_matches(
            glob.glob(str(experiment_dir / "[1-9]*"))[0],
            f"data/testing/reference_experiments/bc_human_{bc_human_name}",
        )
        bc_human_results[bc_human_name] = bc_human_result

    # TODO: test alphazero_assistant
    # TODO: test ppo_assistant
    # TODO: test human_rollout
    # TODO: test masked_bc_human
    # TODO: test bc_assistant
