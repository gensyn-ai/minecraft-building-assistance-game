import copy
import random
from typing import cast

import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.mbag_env import DEFAULT_CONFIG, MbagEnv
from mbag.environment.types import MbagActionTuple


def _convert_state(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [_convert_state(el) for el in data]
    elif isinstance(data, tuple):
        return tuple(_convert_state(el) for el in data)
    elif isinstance(data, dict):
        return {key: _convert_state(value) for key, value in data.items()}
    elif hasattr(data, "__dict__"):
        return _convert_state(data.__dict__)
    else:
        return data


def test_config_not_changed():
    """The env config dictionary passed should not be modified by the environment."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    MbagEnv(config)
    assert config == DEFAULT_CONFIG


def test_deterministic():
    for teleporation in [False, True]:
        for inf_blocks in [False, True]:
            horizon = 100
            env_a = MbagEnv(
                {
                    "goal_generator": "random",
                    "horizon": horizon,
                    "abilities": {
                        "teleportation": teleporation,
                        "flying": True,
                        "inf_blocks": inf_blocks,
                    },
                    "world_size": (5, 5, 5),
                }
            )
            (obs,) = env_a.reset()
            env_b = MbagEnv(env_a.config)
            env_b.reset()
            action_map = MbagActionDistribution.get_action_mapping(env_a.config)

            for t in range(horizon):
                env_b.set_state(env_a.get_state())
                obs_batch = obs[0][None], obs[1][None], obs[2][None]
                mask = MbagActionDistribution.get_mask_flat(env_a.config, obs_batch)[0]
                flat_action = random.choice(mask.nonzero()[0])
                action = cast(MbagActionTuple, tuple(action_map[flat_action]))
                (obs,), rewards_a, _, _ = env_a.step([action])
                state_after_a = env_a.get_state()
                (obs,), rewards_b, _, _ = env_b.step([action])
                state_after_b = env_b.get_state()
                assert _convert_state(state_after_a) == _convert_state(state_after_b)
                assert rewards_a == rewards_b
