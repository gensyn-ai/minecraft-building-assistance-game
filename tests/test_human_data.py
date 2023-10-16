import tempfile

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.offline import JsonReader
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import ENV_CREATOR, _global_registry

import mbag.rllib  # noqa: F401
from mbag.environment.mbag_env import MbagConfigDict
from mbag.scripts.convert_human_data_to_rllib import (
    ex as convert_human_data_to_rllib_ex,
)


def test_convert_human_data_to_rllib():
    for flat_actions, env_id in [
        (False, "MBAG-v1"),
        (True, "MBAGFlatActions-v1"),
    ]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = convert_human_data_to_rllib_ex.run(
                config_updates={
                    "data_dir": "data/human_data/sample_tutorial",
                    "flat_actions": flat_actions,
                    "flat_observations": False,
                    "out_dir": tmp_dir,
                }
            ).result
            reader = JsonReader(tmp_dir)
            episode = reader.next()

            assert result is not None
            mbag_config: MbagConfigDict = result["mbag_config"]
            mbag_config["malmo"]["use_malmo"] = False
            env: MultiAgentEnv = _global_registry.get(ENV_CREATOR, env_id)(mbag_config)

            obs_dict, info_dict = env.reset()
            for t in range(len(episode)):
                for obs_piece, expected_obs_piece in zip(
                    obs_dict["player_0"][:2], episode[SampleBatch.OBS][t][:2]
                ):
                    assert np.all(obs_piece == expected_obs_piece)
                (
                    obs_dict,
                    reward_dict,
                    terminated_dict,
                    truncated_dict,
                    info_dict,
                ) = env.step({"player_0": episode[SampleBatch.ACTIONS][t]})
            assert terminated_dict["player_0"]
            assert info_dict["player_0"]["goal_similarity"] == 216
