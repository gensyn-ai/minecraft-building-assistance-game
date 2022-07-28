from typing import TypedDict, Union
import gym
import numpy as np
from gym import spaces
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.mbag_env import MbagConfigDict, MbagStateDict
from mbag.environment.types import MbagObs
from .rllib_env import FlatActionSpaceWrapper, MbagMultiAgentEnv, MbagRllibWrapper


class MbagObsWithMask(TypedDict):
    obs: MbagObs
    action_mask: np.ndarray


class MbagEnvModelStateDict(MbagStateDict, total=False):
    last_obs: Union[MbagObs, MbagObsWithMask]


class MbagEnvModel(gym.Env):
    """
    A single-agent environment model that can be used for planning in algorithms such
    as AlphaZero.
    """

    action_space: spaces.Discrete
    last_obs: Union[MbagObs, MbagObsWithMask]

    def __init__(
        self,
        env: MbagRllibWrapper,
        config: MbagConfigDict,
        player_index: int = 0,
        include_action_mask_in_obs=True,
    ):
        super().__init__()

        self.env = env
        self.config = config
        self.player_index = player_index
        self.agent_id = f"player_{player_index}"
        self.include_action_mask_in_obs = include_action_mask_in_obs

        assert isinstance(self.env.action_space, spaces.Discrete)
        self.action_space = self.env.action_space
        if include_action_mask_in_obs:
            self.observation_space = spaces.Dict(
                {
                    "obs": self.env.observation_space,
                    "action_mask": spaces.Box(
                        np.zeros(self.action_space.n, dtype=bool),
                        np.ones(self.action_space.n, dtype=bool),
                    ),
                }
            )
        else:
            self.observation_space = self.env.observation_space

    def _store_last_obs_dict(self, obs_dict: MultiAgentDict):
        # For now, don't store, just make sure there aren't any other players.
        if set(obs_dict.keys()) != {self.agent_id}:
            raise RuntimeError(
                f"Expected just {self.agent_id} but received {', '.join(obs_dict.keys())}"
            )

    def _process_obs(self, obs: MbagObs):
        if self.include_action_mask_in_obs:
            world_obs, inventory_obs, timestep = obs
            self.last_obs = {
                "obs": obs,
                "action_mask": MbagActionDistribution.get_mask_flat(
                    self.config,
                    (world_obs[None], inventory_obs[None], timestep[None]),
                )[0],
            }
        else:
            self.last_obs = obs
        return self.last_obs

    def reset(self):
        obs_dict = self.env.reset()
        self._store_last_obs_dict(obs_dict)
        return self._process_obs(obs_dict[self.agent_id])

    def step(self, action):
        action_dict = {self.agent_id: action}
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        self._store_last_obs_dict(obs_dict)
        return (
            self._process_obs(obs_dict[self.agent_id]),
            reward_dict[self.agent_id],
            done_dict.get(self.agent_id, done_dict["__all__"]),
            info_dict[self.agent_id],
        )

    def get_state(self) -> MbagEnvModelStateDict:
        env_state = self.env.get_state()
        return {
            "current_blocks": env_state["current_blocks"],
            "goal_blocks": env_state["goal_blocks"],
            "player_locations": env_state["player_locations"],
            "player_directions": env_state["player_directions"],
            "player_inventories": env_state["player_inventories"],
            "last_interacted": env_state["last_interacted"],
            "timestep": env_state["timestep"],
            "last_obs": self.last_obs,
        }

    def set_state(self, state: MbagEnvModelStateDict):
        if "last_obs" in state:
            self.env.set_state_no_obs(state)
            self.last_obs = state["last_obs"]
            # TODO: load last obs dict?
            return self.last_obs
        else:
            obs_dict = self.env.set_state(state)
            self._store_last_obs_dict(obs_dict)
            return self._process_obs(obs_dict[self.agent_id])


def create_mbag_env_model(config: MbagConfigDict):
    # We should never use Malmo in the env model.
    config["malmo"]["use_malmo"] = False
    env = MbagMultiAgentEnv(config)
    env = FlatActionSpaceWrapper(env, config)
    env = MbagEnvModel(env, config)
    return env


register_env("MBAGAlphaZeroModel-v1", create_mbag_env_model)
