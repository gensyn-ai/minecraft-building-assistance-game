from typing import Dict, List, Tuple, Type, cast
import gym
import torch
from torch import nn
import numpy as np
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.tune.registry import register_trainable

from mbag.environment.types import MbagAction, MbagActionTuple, MbagObs
from mbag.agents.mbag_agent import MbagAgent
from .torch_action_distributions import MbagAutoregressiveActionDistribution
from mbag.environment.types import GOAL_BLOCKS
from mbag.rllib.torch_models import MbagTorchModel


class MbagAgentPolicy(Policy):
    """
    An RLlib policy that selects actions based on an MBAG agent instance.
    """

    agent: MbagAgent

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict,
    ):
        super().__init__(observation_space, action_space, config)
        self.agent = config["mbag_agent"]
        self.exploration = self._create_exploration()

    def get_initial_state(self) -> List[TensorType]:
        self.agent.reset()
        return self.agent.get_state()

    def compute_actions(
        self,
        obs_batch: List[np.ndarray],
        state_batches: List[np.ndarray],
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        for batch_key, view_requirement in self.view_requirements.items():
            if batch_key.startswith("state_in_"):
                view_requirement.batch_repeat_value = 1

        unflattened_obs_batch = restore_original_dimensions(
            obs_batch,
            obs_space=self.observation_space,
            tensorlib=np,
        )

        actions: List[MbagActionTuple] = []
        new_states: List[List[np.ndarray]] = []

        obs: MbagObs
        prev_state: Tuple[np.ndarray, ...]
        for obs, prev_state in zip(
            zip(*unflattened_obs_batch),
            zip(*state_batches)
            if state_batches != []
            else [[] for _ in range(len(obs_batch))],
        ):
            self.agent.set_state(list(prev_state))
            action = self.agent.get_action(obs)
            actions.append(action)
            new_states.append(self.agent.get_state())

        action_arrays = tuple(
            np.array([action[action_part] for action in actions])
            for action_part in range(3)
        )
        state_arrays = [
            np.array([new_state[state_part] for new_state in new_states])
            for state_part in range(len(state_batches))
        ]
        return action_arrays, state_arrays, {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


class MbagPPOTorchPolicy(PPOTorchPolicy):
    def loss(
        self,
        model: MbagTorchModel,
        dist_class: Type[MbagAutoregressiveActionDistribution],
        train_batch: SampleBatch,
    ) -> TensorType:
        loss = super().loss(model, dist_class, train_batch)
        assert not isinstance(loss, list)

        (world_obs,) = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=self.observation_space,
            tensorlib=torch,
        )

        goal = world_obs[:, GOAL_BLOCKS].long()

        loss += self.config["place_block_loss_coeff"] * self.place_block_loss(
            model, dist_class, goal, train_batch
        )
        loss += self.config["goal_loss_coeff"] * self.predict_goal_loss(
            model, goal, train_batch
        )

        return loss

    def predict_goal_loss(
        self,
        model: MbagTorchModel,
        goal: TensorType,
        train_batch: SampleBatch,
    ) -> TensorType:
        if not hasattr(model, "_backbone_out"):
            model(train_batch)
        log_odds = model.goal_function()

        ce = nn.CrossEntropyLoss()
        loss = ce(log_odds, goal)

        model.tower_stats["predict_goal_loss"] = loss

        return loss

    def place_block_loss(
        self,
        model: MbagTorchModel,
        dist_class: Type[MbagAutoregressiveActionDistribution],
        goal: TensorType,
        train_batch,
    ) -> TensorType:
        """
        Add loss to minimize the cross-entropy between the block ID for a "place block" action
        and the goal block at that location, if there is any goal block there.
        """

        actions = train_batch[SampleBatch.ACTIONS].long()

        place_block_ids = goal.flatten(start_dim=1)[
            torch.arange(0, actions.size()[0], dtype=torch.long, device=goal.device),
            actions[:, 1],
        ]

        logits = model.logits if hasattr(model, "logits") else model(train_batch)[0]

        action_dist = dist_class(logits, model)
        place_block_loss = -action_dist._block_id_distribution(
            actions[:, 0],
            actions[:, 1],
        ).logp(place_block_ids)

        # We only care about place block actions at places where there are blocks in the
        # goal.
        place_block_mask = (actions[:, 0] == MbagAction.PLACE_BLOCK) & ~torch.any(
            place_block_ids[:, None]
            == MbagAutoregressiveActionDistribution.PLACEABLE_BLOCK_MASK[None, :],
            dim=1,
        )
        if torch.any(place_block_mask):
            place_block_loss = place_block_loss[place_block_mask]
            if self.config["sum_loss"]:
                place_block_loss = place_block_loss.sum()
            else:
                place_block_loss = place_block_loss.mean()
            model.tower_stats["place_block_loss"] = place_block_loss
            return place_block_loss
        else:
            return 0

    def log_mean_loss(self, info: Dict[str, TensorType], loss_name: str):
        try:
            info[loss_name] = torch.mean(torch.stack(self.get_tower_stats(loss_name)))
        except AssertionError:
            info[loss_name] = torch.nan

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        info = super().extra_grad_info(train_batch)

        self.log_mean_loss(info, "place_block_loss")
        self.log_mean_loss(info, "predict_goal_loss")

        return cast(
            Dict[str, TensorType],
            convert_to_numpy(info),
        )


class MbagPPOTrainer(PPOTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return {
            **super().get_default_config(),
            "goal_loss_coeff": 1.0,
            "place_block_loss_coeff": 1.0,
            "sum_loss": False,
        }

    def get_default_policy_class(self, config):
        return MbagPPOTorchPolicy


register_trainable("MbagPPO", MbagPPOTrainer)
