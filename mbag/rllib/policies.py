from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.torch_policy import TorchPolicy
from mbag.environment.blocks import MinecraftBlocks
from typing import Callable, Dict, List, Tuple, Type
import gym
import numpy as np
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions

# TODO: update to newer RLlib interface
from ray.rllib.agents.ppo.ppo_torch_policy import (  # type: ignore
    PPOTorchPolicy,
    ppo_surrogate_loss,
    kl_and_loss_stats as ppo_stats,
)
from ray.rllib.agents.ppo.appo_torch_policy import (
    AsyncPPOTorchPolicy,
    appo_surrogate_loss,
    stats as appo_stats,
)
from ray.rllib.agents.impala.vtrace_torch_policy import (
    VTraceTorchPolicy,
    build_vtrace_loss,
    stats as vtrace_stats,
)

from mbag.environment.types import MbagAction, MbagActionTuple, MbagObs
from mbag.agents.mbag_agent import MbagAgent
from .torch_action_distributions import MbagAutoregressiveActionDistribution


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
        self.view_requirements["state_in_0"].batch_repeat_value = 1

        unflattened_obs_batch = restore_original_dimensions(
            obs_batch,
            obs_space=self.observation_space,
            tensorlib=np,
        )

        actions: List[MbagActionTuple] = []
        new_states: List[List[np.ndarray]] = []

        obs: MbagObs
        prev_state: Tuple[np.ndarray, ...]
        for obs, prev_state in zip(zip(*unflattened_obs_batch), zip(*state_batches)):
            self.agent.set_state(list(prev_state))
            action = self.agent.get_action(obs)
            actions.append(action)
            new_states.append(self.agent.get_state())

        action_arrays = tuple(
            np.array([action[action_part] for action in actions])
            for action_part in range(3)
        )
        state_arrays = [
            np.array([new_state[state_part] for new_state in new_states], dtype=float)
            for state_part in range(len(state_batches))
        ]
        return action_arrays, state_arrays, {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


def add_supervised_loss_to_policy(
    policy_class: Type[TorchPolicy],
    name: str,
    loss_fn: Callable[
        [Policy, ModelV2, Type[TorchDistributionWrapper], SampleBatch], TensorType
    ],
    stats_fn: Callable[[Policy, SampleBatch], Dict[str, TensorType]],
    sum_loss: bool = False,
) -> Type[TorchPolicy]:
    """
    Adds a supervised loss to the existing policy which minimizes the cross-entropy
    between the block ID for a "place block" action and the goal block at that location,
    if there is any goal block there.
    """

    def loss_with_supervision(
        policy: Policy,
        model: TorchModelV2,
        dist_class: Type[MbagAutoregressiveActionDistribution],
        train_batch: SampleBatch,
    ) -> TensorType:

        import torch

        loss = loss_fn(policy, model, dist_class, train_batch)
        assert not isinstance(loss, list)

        (world_obs,) = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=policy.observation_space,
            tensorlib=torch,
        )
        goal_block_ids = world_obs[:, 2]

        actions = train_batch[SampleBatch.ACTIONS].long()
        place_block_ids = goal_block_ids.flatten(start_dim=1)[
            torch.arange(
                0, actions.size()[0], dtype=torch.long, device=goal_block_ids.device
            ),
            actions[:, 1],
        ]
        logits, state = model.from_batch(train_batch, is_training=True)
        action_dist = dist_class(logits, model)
        place_block_loss = -action_dist._block_id_distribution(
            actions[:, 0],
            actions[:, 1],
        ).logp(place_block_ids)

        # We only care about place block actions at places where there are blocks in the
        # goal.
        place_block_mask = (actions[:, 0] == MbagAction.PLACE_BLOCK) & (
            place_block_ids != MinecraftBlocks.AIR
        )
        if torch.any(place_block_mask):
            place_block_loss = place_block_loss[place_block_mask]
            if sum_loss:
                place_block_loss = place_block_loss.sum()
            else:
                place_block_loss = place_block_loss.mean()
            loss = loss + place_block_loss
            policy._place_block_loss = place_block_loss  # type: ignore

        return loss

    def supervision_stats(
        policy: TorchPolicy, train_batch: SampleBatch
    ) -> Dict[str, TensorType]:
        stats = stats_fn(policy, train_batch)
        if hasattr(policy, "_place_block_loss"):
            stats["place_block_loss"] = policy._place_block_loss  # type: ignore
        return stats

    return policy_class.with_updates(  # type: ignore
        name=name,
        loss_fn=loss_with_supervision,
        stats_fn=supervision_stats,
    )


MbagPPOTorchPolicy = add_supervised_loss_to_policy(
    PPOTorchPolicy,
    name="MbagPPOTorchPolicy",
    loss_fn=ppo_surrogate_loss,
    stats_fn=ppo_stats,
)

MbagAPPOTorchPolicy = add_supervised_loss_to_policy(
    AsyncPPOTorchPolicy,
    name="MbagAPPOTorchPolicy",
    loss_fn=appo_surrogate_loss,
    stats_fn=appo_stats,
)

MbagVTraceTorchPolicy = add_supervised_loss_to_policy(
    VTraceTorchPolicy,
    name="MbagVTraceTorchPolicy",
    loss_fn=build_vtrace_loss,
    stats_fn=vtrace_stats,
    sum_loss=True,
)


MBAG_POLICIES = {
    "PPO": MbagPPOTorchPolicy,
    "APPO": MbagAPPOTorchPolicy,
    "IMPALA": MbagVTraceTorchPolicy,
}
