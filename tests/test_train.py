import pytest
import numpy as np
from  mbag.rllib.train import ex 
import tempfile
import random
import time

# Make temporary file that will be deleted afterward
tempdir = tempfile.mkdtemp()

# This seeds the random number generator with the current nanosecond time.
@pytest.fixture
def random_seed():
    random.seed(time.time_ns())


def test_single_agent():
    result = ex.run(
        config_updates={
            "log_dir": tempdir,
            "width": 10,
            "depth": 10,
            "height": 10,
            "kl_target": .01,
            "horizon": 1,
            "num_workers": 16,
            "goal_generator": 'single_wall_grabcraft',
            "use_extra_features": True,
            "num_training_iters": 10,
            "train_batch_size": 100,
            "sgd_minibatch_size": 10,
        }).result


    # print(result["info"]["learner"]["ppo"].keys())
    # print(result["info"]["learner"]["ppo"]['learner_stats'].keys())
    print(result["custom_metrics"])
    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10

def test_transformer():
    transformer_config = {
        "log_dir": tempdir,
        "width": 10,
        "depth": 10,
        "height": 10,
        "kl_target": .01,
        "horizon": 10,
        "num_workers": 16,
        "goal_generator": 'single_wall_grabcraft',
        "use_extra_features": True,
        "num_training_iters": 10,
        "train_batch_size": 1,
        "sgd_minibatch_size": 1,
        "model": "transformer",
        "position_embedding_size": 18,
        "hidden_size": 48,
        "num_layers": 3,
        "num_heads": 2,
    }
    
    result = ex.run(
        config_updates={
            **transformer_config,
            "use_separated_transformer": True     
        }).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10

    # result = ex.run(
    #     config_updates={
    #         **transformer_config,
    #         "num_layers":1,
    #         "use_separated_transformer": False,     
    #     }).result

    # assert result["custom_metrics"]["ppo/own_reward_mean"] > -10

def test_cross_play():
    result = ex.run(
        config_updates={
            "log_dir": tempdir,
            "width": 10,
            "depth": 10,
            "height": 10,
            "kl_target": .01,
            "horizon": 1,
            "num_workers": 16,
            "goal_generator": 'single_wall_grabcraft',
            "use_extra_features": True,
            "num_training_iters": 10,
            "train_batch_size": 100,
            "sgd_minibatch_size": 10,
        }).result


"""
other tests:
    - use transformer
    - train both concurrently
    - distillation
    - retrieve a saved policy and train that
    - self play
"""