from typing import cast

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.offline import JsonReader
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import ENV_CREATOR, _global_registry

import mbag.rllib  # noqa: F401
from mbag.environment.mbag_env import MbagConfigDict
from mbag.rllib.convert_human_data_to_rllib import ex as convert_human_data_to_rllib_ex
from mbag.rllib.human_data import PARTICIPANT_ID


def test_convert_human_data_consistency_with_rllib_env(tmp_path):
    for flat_actions, env_id in [
        (False, "MBAG-v1"),
        (True, "MBAGFlatActions-v1"),
    ]:
        out_dir = str(tmp_path / f"rllib_flat_{flat_actions}")
        result = convert_human_data_to_rllib_ex.run(
            config_updates={
                "data_dir": "data/human_data/sample_tutorial/participant_1",
                "flat_actions": flat_actions,
                "flat_observations": False,
                "out_dir": str(out_dir),
                "offset_rewards": True,
            }
        ).result
        reader = JsonReader(out_dir)
        episode = reader.next()

        assert result is not None
        mbag_config = cast(MbagConfigDict, dict(result["mbag_config"]))
        mbag_config["malmo"]["use_malmo"] = False
        mbag_config["goal_generator"] = "tutorial"
        mbag_config["world_size"] = (6, 6, 6)
        mbag_config["random_start_locations"] = False
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
            assert reward_dict["player_0"] == episode[SampleBatch.REWARDS][t]

        assert terminated_dict["player_0"]
        assert info_dict["player_0"]["goal_similarity"] == 216


def test_convert_human_data_to_rllib_participant_id(tmp_path):
    out_dir = str(tmp_path / "rllib")
    convert_human_data_to_rllib_ex.run(
        config_updates={
            "data_dir": "data/human_data/sample_tutorial",
            "flat_observations": False,
            "out_dir": str(out_dir),
            "offset_rewards": True,
        }
    )
    reader = JsonReader(out_dir)
    for episode_index in range(4):
        episode = reader.next()

        expected_participant_id = episode_index + 1
        assert np.all(episode[PARTICIPANT_ID] == expected_participant_id)
