import glob
import os
import tempfile

import pytest

from mbag.rllib.train import ex


@pytest.fixture(scope="session")
def default_config():
    return {
        "log_dir": tempfile.mkdtemp(),
        "width": 6,
        "depth": 6,
        "height": 6,
        "kl_target": 0.01,
        "horizon": 10,
        "num_workers": 10,
        "goal_generator": "random",
        "use_extra_features": True,
        "num_training_iters": 2,
        "train_batch_size": 50,
        "sgd_minibatch_size": 5,
        "rollout_fragment_length": 10,
    }


@pytest.fixture(scope="session")
def dummy_ppo_checkpoint_fname(default_config):
    # Execute short dummy run and return the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(config_updates={**default_config, "log_dir": checkpoint_dir})

    checkpoint_fname = glob.glob(
        checkpoint_dir
        + "/MbagPPO/self_play/6x6x6/random/*/checkpoint_000002/checkpoint-2"
    )[0]
    assert os.path.exists(checkpoint_fname)
    return checkpoint_fname


@pytest.mark.uses_rllib
def test_single_agent(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "num_training_iters": 10,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_lstm(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "use_per_location_lstm": True,
            "max_seq_len": 5,
            "sgd_minibatch_size": 20,
            "vf_share_layers": True,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_transformer(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "model": "transformer",
            "position_embedding_size": 6,
            "hidden_size": 50,
            "num_layers": 3,
            "num_heads": 1,
            "use_separated_transformer": True,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10

    result = ex.run(
        config_updates={
            **default_config,
            "model": "transformer",
            "position_embedding_size": 6,
            "hidden_size": 50,
            "num_layers": 1,
            "num_heads": 1,
            "use_separated_transformer": False,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_cross_play(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "own_reward_prop": 1,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"ppo": "ppo_0"},
            "policies_to_train": ["ppo_1"],
        }
    ).result

    assert result["custom_metrics"]["ppo_1/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_policy_retrieval(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_path": dummy_ppo_checkpoint_fname,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_distillation(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "run": "distillation_prediction",
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10

    result = ex.run(
        config_updates={
            **default_config,
            "run": "distillation_prediction",
            "heuristic": "mirror_builder",
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_train_together(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "load_policies_mapping": {"ppo": "ppo_0"},
            "policies_to_train": ["ppo_0", "ppo_1"],
        }
    ).result
    assert result["custom_metrics"]["ppo_0/own_reward_mean"] > -10
    assert result["custom_metrics"]["ppo_1/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_alpha_zero(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "run": "MbagAlphaZero",
            "goal_generator": "random",
            "use_replay_buffer": False,
            "hidden_size": 64,
        }
    ).result
    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_alpha_zero_assistant(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "run": "MbagAlphaZero",
            "goal_generator": "random",
            "use_replay_buffer": False,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"ppo": "ppo_0"},
            "policies_to_train": ["ppo_1"],
            "model": "transformer_alpha_zero",
            "hidden_size": 64,
        }
    ).result
    assert result["custom_metrics"]["ppo_0/own_reward_mean"] > -10
    assert result["custom_metrics"]["ppo_1/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_lstm_alpha_zero_assistant(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "run": "MbagAlphaZero",
            "goal_generator": "random",
            "use_replay_buffer": False,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"ppo": "ppo_0"},
            "policies_to_train": ["ppo_1"],
            "model": "transformer_alpha_zero",
            "hidden_size": 64,
            "use_per_location_lstm": True,
            "max_seq_len": 5,
            "sgd_minibatch_size": 20,
            "vf_share_layers": True,
        }
    ).result
    assert result["custom_metrics"]["ppo_0/own_reward_mean"] > -10
    assert result["custom_metrics"]["ppo_1/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_alpha_zero_assistant_with_lowest_block_agent(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "run": "MbagAlphaZero",
            "goal_generator": "random",
            "use_replay_buffer": False,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "policies_to_train": ["ppo_0"],
            "model": "transformer_alpha_zero",
            "hidden_size": 64,
            "heuristic": "lowest_block",
        }
    ).result
    assert result["custom_metrics"]["ppo_0/own_reward_mean"] > -10
    assert result["custom_metrics"]["lowest_block/place_block_accuracy_mean"] == 1
