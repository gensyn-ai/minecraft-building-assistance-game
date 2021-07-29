import argparse
from typing import List, Tuple
import yaml

import ray
from ray import tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.registry import get_trainable_cls


def sample_entropy_coeff_schedule(config) -> List[Tuple[int, float]]:
    start_coeff: float = tune.loguniform(1e-5, 1e-1).sample()
    end_steps = int(tune.loguniform(10, 1e7).sample())
    return [
        (0, start_coeff),
        (end_steps, 0),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_fname", type=str, help="YAML config file")
    args = parser.parse_args()

    with open(args.config_fname, "rb") as config_file:
        all_config = yaml.safe_load(config_file)

    config = all_config["config"]

    config.update(
        {
            "train_batch_size": tune.qrandint(1000, 20000, 1000),
            "sgd_minibatch_size": tune.qrandint(1000, 4000, 1000),
            "num_sgd_iter": tune.randint(1, 10),
            "gamma": tune.uniform(0.9, 1),
            "lr": tune.loguniform(1e-5, 1e-1),
            "kl_target": tune.loguniform(0.001, 1),
            "grad_clip": 1,
            "vf_loss_coeff": tune.loguniform(1e-5, 1e-2),
            "entropy_coeff_schedule": tune.sample_from(sample_entropy_coeff_schedule),
            "evaluation_interval": None,
        }
    )

    if "custom_model_config" not in config:
        config["model"]["custom_model_config"] = {}

    config["model"]["custom_model_config"].update(
        {
            "embedding_size": tune.qrandint(4, 16, 4),
            "num_layers": tune.randint(1, 5),
            "filter_size": tune.choice([3, 5]),
            "hidden_channels": tune.qrandint(8, 64, 8),
        }
    )

    ray.init(ignore_reinit_error=True)

    scheduler = ASHAScheduler(**all_config["scheduler_params"])
    analysis = tune.run(
        get_trainable_cls(all_config["run"]),
        scheduler=scheduler,
        config=config,
        **all_config["tune_params"],
    )
