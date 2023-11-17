import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.alpha_zero.alpha_zero import AlphaZero, AlphaZeroConfig
from ray.rllib.algorithms.alpha_zero.alpha_zero_policy import AlphaZeroPolicy
from ray.rllib.algorithms.alpha_zero.mcts import MCTS, Node, RootParentNode
from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models import ActionDistribution, ModelCatalog, ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import PiecewiseSchedule, Schedule
from ray.rllib.utils.torch_utils import explained_variance, sequence_mask
from ray.rllib.utils.typing import AgentID, PolicyID, ResultDict, TensorType
from ray.tune.registry import ENV_CREATOR, _global_registry, register_trainable
from torch import nn

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import (
    CURRENT_BLOCKS,
    GOAL_BLOCKS,
    MbagAction,
    MbagActionTuple,
)

from .planning import MbagEnvModel, MbagEnvModelInfoDict
from .rllib_env import unwrap_mbag_env
from .torch_models import MbagTorchModel, OtherAgentActionPredictorMixin

MCTS_POLICIES = "mcts_policies"
OTHER_AGENT_ACTION_DIST_INPUTS = "other_agent_action_dist_inputs"


logger = logging.getLogger(__name__)


class MbagRootParentNode(RootParentNode):
    def __init__(self, env):
        super().__init__(env)
        self.action_mapping = MbagActionDistribution.get_action_mapping(
            unwrap_mbag_env(self.env).config
        )


class MbagMCTSNode(Node):
    env: MbagEnvModel
    mcts: "MbagMCTS"
    children: Dict[Tuple[int, ...], "MbagMCTSNode"]
    action_mapping: np.ndarray
    parent: Union["MbagMCTSNode", MbagRootParentNode]
    goal_logits: Optional[np.ndarray]
    other_agent_action_dist: Optional[np.ndarray]
    other_reward: float
    model_state_in: List[np.ndarray]
    model_state_out: List[np.ndarray]

    def __init__(
        self,
        action,
        obs,
        done,
        info: Optional[MbagEnvModelInfoDict],
        reward,
        state,
        mcts,
        model_state_in,
        parent=None,
    ):
        super().__init__(action, obs, done, reward, state, mcts, parent)

        self.info = info

        self.min_value = np.inf
        self.max_value = -np.inf
        self.action_mapping = self.parent.action_mapping

        self.action_type_dirichlet_noise = None
        self.dirichlet_noise = None

        self.goal_logits = None
        self.other_agent_action_dist = None

        self.model_state_in = model_state_in

    def child_Q(self):  # noqa: N802
        Q = self.child_total_value / np.maximum(  # noqa: N806
            self.child_number_visits, 1
        )
        V = (  # noqa: N806
            self.total_value / self.number_visits if self.number_visits > 0 else 0
        )
        Q[self.child_number_visits == 0] = V
        Q = (Q - self.min_value) / max(  # noqa: N806
            self.max_value - self.min_value, 0.01
        )
        return Q

    def action_type_Q(self):  # noqa: N802
        total_value = np.bincount(
            self.action_mapping[:, 0], weights=self.child_total_value
        )
        number_visits = np.bincount(
            self.action_mapping[:, 0], weights=self.child_number_visits
        )
        Q = total_value / np.maximum(number_visits, 1)  # noqa: N806
        V = (  # noqa: N806
            self.total_value / self.number_visits if self.number_visits > 0 else 0
        )
        Q[number_visits == 0] = V
        Q = (Q - self.min_value) / max(  # noqa: N806
            self.max_value - self.min_value, 0.01
        )
        return Q

    def action_type_U(self):  # noqa: N802
        number_visits: np.ndarray = np.bincount(
            self.action_mapping[:, 0], weights=self.child_number_visits
        )
        priors = np.bincount(self.action_mapping[:, 0], weights=self.child_priors)
        return (
            np.sqrt(1 + np.sum(self.child_number_visits)) * priors / (1 + number_visits)
        )

    @property
    def valid_action_types(self) -> np.ndarray:
        return cast(
            np.ndarray,
            np.bincount(
                self.action_mapping[:, 0], weights=self.valid_actions.astype(np.int32)
            )
            > 0,
        )

    def best_action(self) -> int:
        action_type_score = (
            self.action_type_Q() + self.mcts.c_puct * self.action_type_U()
        )
        action_type_score[~self.valid_action_types] = -np.inf
        action_type = np.argmax(action_type_score)

        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[~self.valid_actions] = -np.inf
        masked_child_score[self.action_mapping[:, 0] != action_type] = -np.inf
        return int(np.argmax(masked_child_score))

    def expand(
        self,
        child_priors,
        goal_logits: Optional[np.ndarray] = None,
        other_agent_action_dist: Optional[np.ndarray] = None,
        model_state_out: List[np.ndarray] = [],
        add_dirichlet_noise=False,
    ) -> None:
        super().expand(child_priors)

        self.model_state_out = model_state_out
        self.goal_logits = goal_logits
        self.other_agent_action_dist = other_agent_action_dist

        if (
            self.info is not None
            and self.goal_logits is not None
            and isinstance(self.parent, MbagMCTSNode)
        ):
            # We need to update the reward for this node based on the new goal_logits.
            self.reward = self.env.get_reward_with_other_agent_actions(
                self.parent.obs["obs"],
                self.info,
                self.goal_logits,
            )

        self.child_priors[~self.valid_actions] = 0
        self.child_priors /= self.child_priors.sum()

        if add_dirichlet_noise:
            num_action_types = self.action_mapping[-1, 0] + 1
            type_masks = np.empty(
                (num_action_types, self.valid_actions.shape[0]), dtype=bool
            )
            for action_type in range(num_action_types):
                type_masks[action_type] = (
                    self.action_mapping[:, 0] == action_type
                ) & self.valid_actions
            valid_action_types = np.any(type_masks, axis=1)

            action_type_dirichlet_noise = np.random.dirichlet(
                np.full(num_action_types, 0.25)
            )
            action_type_dirichlet_noise[~valid_action_types] = 0
            action_type_dirichlet_noise /= action_type_dirichlet_noise.sum()

            for action_type in range(num_action_types):
                type_mask = type_masks[action_type]
                if not np.any(type_mask):
                    continue

                self.child_priors[type_mask] *= (
                    (1 - self.mcts.dir_epsilon) * self.child_priors[type_mask].sum()
                    + self.mcts.dir_epsilon * action_type_dirichlet_noise[action_type]
                ) / self.child_priors[type_mask].sum()

                num_valid_actions = type_mask.astype(int).sum()
                alpha = 10 / num_valid_actions
                dirichlet_noise = np.random.dirichlet(np.full(num_valid_actions, alpha))
                self.child_priors[type_mask] = (
                    1 - self.mcts.dir_epsilon
                ) * self.child_priors[
                    type_mask
                ] + self.mcts.dir_epsilon * self.child_priors[
                    type_mask
                ].sum() * dirichlet_noise

            assert abs(self.child_priors.sum() - 1) < 1e-2

    def get_child(self, action: int) -> "MbagMCTSNode":
        all_actions: Tuple[int, ...] = (action,)
        other_agent_actions: Optional[List[int]] = None
        if self.other_agent_action_dist is not None:
            other_agent_action = np.random.choice(
                np.arange(self.action_space_size), p=self.other_agent_action_dist
            )
            all_actions = action, other_agent_action
            other_agent_actions = [other_agent_action]

        if all_actions not in self.children:
            self.env.set_state(self.state)
            assert self.goal_logits is not None
            obs, reward, terminated, truncated, info = self.env.step(
                action,
                goal_logits=self.goal_logits,
                other_player_actions=other_agent_actions,
            )
            next_state = self.env.get_state()
            self.children[all_actions] = MbagMCTSNode(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=terminated,
                info=info,
                obs=obs,
                mcts=self.mcts,
                model_state_in=self.model_state_out,
            )
        return self.children[all_actions]

    def backup(self, value):
        current = self
        value = float(value)
        while True:
            value *= self.mcts.gamma
            value += current.reward
            current.number_visits += 1
            current.total_value += value
            for node in [current, current.parent]:
                if isinstance(node, MbagMCTSNode):
                    node.min_value = min(node.min_value, value)
                    node.max_value = max(node.max_value, value)
            if isinstance(current.parent, MbagMCTSNode):
                current = current.parent
            else:
                break

    def get_mbag_action(self, flat_action: int) -> MbagAction:
        return MbagAction(
            cast(MbagActionTuple, tuple(self.action_mapping[flat_action])),
            self.env.config["world_size"],
        )


class MbagMCTS(MCTS):
    _temperature_schedule: Optional[Schedule]

    def __init__(self, model, mcts_param, gamma: float, use_critic=True):
        super().__init__(model, mcts_param)
        self.gamma = gamma  # Discount factor.
        self.use_critic = use_critic

        self._temperature_schedule = None
        if mcts_param["temperature_schedule"] is not None:
            self._temperature_schedule = PiecewiseSchedule(
                mcts_param["temperature_schedule"],
                outside_value=mcts_param["temperature_schedule"][-1][-1],
                framework=None,
            )
            self.temperature = self._temperature_schedule.value(0)

    def update_temperature(self, global_timestep: int):
        if self._temperature_schedule is not None:
            self.temperature = self._temperature_schedule.value(global_timestep)

    def compute_action(self, node: MbagMCTSNode):
        for _ in range(self.num_sims):
            leaf: MbagMCTSNode = node.select()
            if leaf.done:
                value = 0
            else:
                self.model.eval()
                child_priors, value, state_out = self.model.compute_priors_and_value(
                    leaf.obs,
                    leaf.model_state_in,
                )

                if not self.use_critic:
                    value = 0

                goal_logits = convert_to_numpy(self.model.goal_function())[0]

                other_agent_action_dist: Optional[np.ndarray] = None
                if isinstance(self.model, OtherAgentActionPredictorMixin):
                    other_agent_action_dist = convert_to_numpy(
                        self.model.predict_other_agent_action().softmax(1)
                    )[0]

                leaf.expand(
                    child_priors,
                    goal_logits=goal_logits,
                    other_agent_action_dist=other_agent_action_dist,
                    add_dirichlet_noise=self.add_dirichlet_noise and leaf == node,
                    model_state_out=state_out,
                )

            leaf.backup(value)

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(
            tree_policy
        )  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)

        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = int(np.argmax(tree_policy))
        else:
            # otherwise sample an action according to tree policy probabilities
            action = int(
                np.random.choice(np.arange(node.action_space_size), p=tree_policy)
            )

        if logger.isEnabledFor(logging.DEBUG):
            child = node.get_child(action)
            logger.debug(
                "\t".join(
                    map(
                        str,
                        [
                            node.get_mbag_action(action),
                            child.reward,
                            node.child_Q()[action],
                            node.child_U()[action],
                            node.child_priors[action],
                            child.number_visits,
                            tree_policy[action],
                            node.valid_actions.astype(int).sum(),
                        ],
                    )
                )
            )

            self.model.eval()
            _, value = self.model.compute_priors_and_value(node.obs)
            logger.debug(f"{value} {node.total_value / node.number_visits}")

            plan = []
            current = node
            while len(current.children) > 0:
                plan_action = int(np.argmax(current.child_number_visits))
                current = current.get_child(plan_action)
                plan.append((node.get_mbag_action(plan_action), current.reward))
            logger.debug(plan)

        return tree_policy, action, node.get_child(action)


class MbagAlphaZeroPolicy(AlphaZeroPolicy, EntropyCoeffSchedule):
    mcts: MbagMCTS
    env: MbagEnvModel
    config: Dict[str, Any]

    def __init__(self, obs_space, action_space, config):
        model = ModelCatalog.get_model_v2(
            obs_space, action_space, action_space.n, config["model"], "torch"
        )

        def env_creator():
            env_creator = _global_registry.get(ENV_CREATOR, config["env"])
            # We should never use Malmo in the env model.
            env_config = copy.deepcopy(config["env_config"])
            env_config["malmo"]["use_malmo"] = False
            env = env_creator(env_config)
            return MbagEnvModel(env, env_config)

        def mcts_creator():
            return MbagMCTS(model, config["mcts_config"], config["gamma"])

        super().__init__(
            obs_space,
            action_space,
            config,
            model=model,
            loss=None,
            action_distribution_class=TorchCategorical,
            mcts_creator=mcts_creator,
            env_creator=env_creator,
        )

        self.mcts.model = self.model

        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )

    def compute_actions_from_input_dict(
        self, input_dict, explore=None, timestep=None, episodes=None, **kwargs
    ):
        assert self.mcts.model == self.model

        model_state_len = sum(k[:8] == "state_in" for k in input_dict.keys())
        state_out = [
            np.empty_like(input_dict[f"state_in_{state_index}"])
            for state_index in range(model_state_len)
        ]
        if self.config["pretrain"]:
            for state_out_part in state_out:
                state_out_part[:] = 0

        player_index = int(input_dict[SampleBatch.AGENT_INDEX])
        self.env.set_player_index(player_index)

        with torch.no_grad():
            actions = []
            expected_rewards: List[float] = []
            expected_own_rewards: List[float] = []
            episode: Episode

            for episode_index, episode in enumerate(episodes):
                if episode.length == 0:
                    episode.user_data[MCTS_POLICIES] = []

                if self.config["pretrain"]:
                    actions.append(0)
                    expected_rewards.append(0)
                    expected_own_rewards.append(0)
                    assert isinstance(self.action_space, spaces.Discrete)
                    mcts_policy = np.zeros((self.action_space.n,))
                    mcts_policy[0] = 1
                    episode.user_data[MCTS_POLICIES].append(mcts_policy)
                    continue

                env_state = episode.user_data["state"]
                model_state = [
                    input_dict[f"state_in_{state_index}"][episode_index]
                    for state_index in range(model_state_len)
                ]
                # verify if env has been wrapped for ranked rewards
                if self.env.__class__.__name__ == "RankedRewardsEnvWrapper":
                    # r2 env state contains also the rewards buffer state
                    env_state = {"env_state": env_state, "buffer_state": None}
                # create tree root node
                obs = self.env.set_state(env_state)
                tree_node = MbagMCTSNode(
                    state=env_state,
                    obs=obs,
                    reward=0,
                    done=False,
                    info=None,
                    action=None,
                    parent=MbagRootParentNode(env=self.env),
                    model_state_in=model_state,
                    mcts=self.mcts,
                )

                # run monte carlo simulations to compute the actions
                # and record the tree
                mcts_policy, action, action_node = self.mcts.compute_action(tree_node)
                # record action
                actions.append(action)

                # Store MCTS policies vectors and other info.
                episode.user_data[MCTS_POLICIES].append(mcts_policy)
                expected_rewards.append(action_node.reward)
                if action_node.info is not None:
                    expected_own_rewards.append(action_node.info["own_reward"])
                else:
                    expected_own_rewards.append(np.nan)

                policy_id = self.config["__policy_id"]
                for metric_id, metric_value in [
                    ("expected_reward", expected_rewards[-1]),
                    ("expected_own_reward", expected_own_rewards[-1]),
                ]:
                    metric_key = f"{policy_id}/{metric_id}"
                    if metric_key not in episode.custom_metrics:
                        episode.custom_metrics[metric_key] = 0
                    episode.custom_metrics[metric_key] += metric_value

                for state_index, state in enumerate(tree_node.model_state_out):
                    state_out[state_index][episode_index] = state

            return (
                np.array(actions),
                state_out,
                {
                    **self.extra_action_out(
                        input_dict,
                        kwargs.get("state_batches", []),
                        self.model,
                        cast(Any, None),
                    ),
                    "expected_reward": np.array(expected_rewards),
                    "expected_own_reward": np.array(expected_own_rewards),
                },
            )

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple[PolicyID, Type[TorchPolicy], SampleBatch]]
        ] = None,
        episode=None,
    ):
        with torch.no_grad():
            last_r = 0
            rewards_plus_v = np.concatenate(
                [sample_batch[SampleBatch.REWARDS], np.array([last_r])]
            )
            discounted_returns = discount_cumsum(rewards_plus_v, self.config["gamma"])[
                :-1
            ].astype(np.float32)
            sample_batch[Postprocessing.VALUE_TARGETS] = discounted_returns

        # Add MCTS policies to sample batch.
        sample_batch[MCTS_POLICIES] = np.array(episode.user_data[MCTS_POLICIES])[
            sample_batch[SampleBatch.T]
        ]

        if other_agent_batches is not None:
            if len(other_agent_batches) > 1:
                raise RuntimeError(
                    "Training with multiple other agents is not supported."
                )
            elif len(other_agent_batches) == 1:
                other_agent_id, (_, _, other_agent_batch) = next(
                    iter(other_agent_batches.items())
                )
                if SampleBatch.ACTION_DIST_INPUTS in other_agent_batch:
                    sample_batch[OTHER_AGENT_ACTION_DIST_INPUTS] = other_agent_batch[
                        SampleBatch.ACTION_DIST_INPUTS
                    ]
                else:
                    logger.warn(
                        f"no action_dist_inputs in sample batch for {other_agent_id}"
                    )
            else:
                pass  # No need to include other agent batch for single player case.

        return sample_batch

    def learn_on_batch(self, postprocessed_batch):
        return TorchPolicy.learn_on_batch(self, postprocessed_batch)

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        assert isinstance(model, MbagTorchModel)

        # Forward pass in model.
        logits, state = model(train_batch)
        values = model.value_function()
        logits, values = torch.squeeze(logits), torch.squeeze(values)
        dist = dist_class(logits, model=model)
        assert isinstance(dist, TorchCategorical)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])  # noqa: N806
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            assert isinstance(mask, torch.Tensor)
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        # Compute actor and critic losses.
        policy_loss = reduce_mean_valid(
            -torch.sum(train_batch[MCTS_POLICIES] * dist.dist.logits, dim=-1)
        )
        value_loss = reduce_mean_valid(
            (values - train_batch[Postprocessing.VALUE_TARGETS]) ** 2
        )

        entropy = reduce_mean_valid(dist.entropy())

        # Compute goal prediction loss.
        goal_logits = model.goal_function()
        world_obs, _, _ = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=self.observation_space,
            tensorlib=torch,
        )
        goal = world_obs[:, GOAL_BLOCKS].long()
        ce = nn.CrossEntropyLoss(reduction="none")
        goal_ce: torch.Tensor = ce(goal_logits, goal)
        goal_loss = reduce_mean_valid(goal_ce.flatten(start_dim=1).mean(dim=1))

        unplaced_blocks = (goal != MinecraftBlocks.AIR) & (
            world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.AIR
        )
        unplaced_blocks_goal_loss = goal_ce[unplaced_blocks].mean()

        # Compute total loss.
        total_loss: torch.Tensor = (
            self.config["vf_loss_coeff"] * value_loss
            + self.config["goal_loss_coeff"] * goal_loss
            - self.entropy_coeff * entropy
        )
        if not self.config["pretrain"]:
            total_loss = total_loss + policy_loss

        if isinstance(model, OtherAgentActionPredictorMixin):
            # Compute other agent action prediction loss.
            predicted_other_agent_action_dist = dist_class(
                cast(Any, model.predict_other_agent_action()),
                model=model,
            )
            other_agent_action_dist_inputs = train_batch[OTHER_AGENT_ACTION_DIST_INPUTS]
            # Get rid of -inf action dist inputs to avoid numeric issues with
            # KL divergence.
            other_agent_action_dist_inputs[
                other_agent_action_dist_inputs == -np.inf
            ] = -1e4
            actual_other_agent_action_dist = dist_class(
                [other_agent_action_dist_inputs],
                model=model,
            )
            other_agent_action_predictor_loss = reduce_mean_valid(
                actual_other_agent_action_dist.kl(predicted_other_agent_action_dist)
            )

            model.tower_stats[
                "other_agent_action_predictor_loss"
            ] = other_agent_action_predictor_loss
            total_loss = (
                total_loss
                + self.config["other_agent_action_predictor_loss_coeff"]
                * other_agent_action_predictor_loss
            )

        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["policy_loss"] = policy_loss
        model.tower_stats["vf_loss"] = value_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], values
        )
        model.tower_stats["goal_loss"] = goal_loss
        model.tower_stats["unplaced_blocks_goal_loss"] = unplaced_blocks_goal_loss
        model.tower_stats["entropy"] = entropy

        return total_loss

    def extra_grad_info(self, train_batch: SampleBatch):
        grad_info: Dict[str, TensorType] = {
            "entropy_coeff": self.entropy_coeff,
            "mcts/temperature": self.mcts.temperature,
        }
        for metric in [
            "total_loss",
            "policy_loss",
            "vf_loss",
            "vf_explained_var",
            "goal_loss",
            "unplaced_blocks_goal_loss",
            "entropy",
            "other_agent_action_predictor_loss",
        ]:
            try:
                grad_info[metric] = torch.mean(
                    torch.stack(cast(List[torch.Tensor], self.get_tower_stats(metric)))
                )
            except AssertionError:
                pass
        return convert_to_numpy(grad_info)

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"]
            )
        self.mcts.update_temperature(global_timestep=global_vars["timestep"])


class MbagAlphaZeroConfig(AlphaZeroConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class)

        self.sample_batch_size = 1000
        self.vf_loss_coeff = 1.0
        self.other_agent_action_predictor_loss_coeff = 1.0
        self.goal_loss_coeff = 1.0
        self.entropy_coeff = 0
        self.entropy_coeff_schedule = 0
        self.use_critic = True
        self.use_replay_buffer = True
        self.num_steps_sampled_before_learning_starts = 0
        self.pretrain = False

        del self.vf_share_layers
        # self.mcts_config["temperature_schedule"] = None

    def training(
        self,
        *args,
        sample_batch_size=NotProvided,
        vf_loss_coeff=NotProvided,
        other_agent_action_predictor_loss_coeff=NotProvided,
        goal_loss_coeff=NotProvided,
        entropy_coeff=NotProvided,
        entropy_coeff_schedule=NotProvided,
        use_critic=NotProvided,
        use_replay_buffer=NotProvided,
        num_steps_sampled_before_learning_starts=NotProvided,
        pretrain=NotProvided,
        **kwargs,
    ):
        """
        Set training parameters.
        Args:
            sample_batch_size (int): Number of samples to include in each
                training batch.
            vf_loss_coeff (float): Coefficient of the value function loss.
            other_agent_action_predictor_loss_coeff (float): Coefficient of the
                other agent action predictor loss.
            goal_loss_coeff (float): Coefficient of the goal predictor loss.
            entropy_coeff (float): Coefficient of the entropy loss.
            entropy_coeff_schedule (float): Schedule for the entropy
                coefficient.
            use_critic (bool): Whether to use a critic.
            use_replay_buffer (bool): Whether to use a replay buffer.
            num_steps_sampled_before_learning_starts (int): Number of steps
                collected before learning starts.
            pretrain (bool): If True, then this will just pretrain the AlphaZero
                predictors for goal, other agent action, etc. and take only NOOP
                actions.
        """

        super().training(*args, **kwargs)

        if sample_batch_size is not NotProvided:
            self.sample_batch_size = sample_batch_size
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if other_agent_action_predictor_loss_coeff is not NotProvided:
            self.other_agent_action_predictor_loss_coeff = (
                other_agent_action_predictor_loss_coeff
            )
        if goal_loss_coeff is not NotProvided:
            self.goal_loss_coeff = goal_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if use_critic is not NotProvided:
            self.use_critic = use_critic
        if use_replay_buffer is not NotProvided:
            self.use_replay_buffer = use_replay_buffer
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )
        if pretrain is not NotProvided:
            self.pretrain = pretrain

    def update_from_dict(self, config_dict):
        if "mcts_config" in config_dict and isinstance(config_dict, dict):
            self.mcts_config.update(config_dict["mcts_config"])
            del config_dict["mcts_config"]

        return super().update_from_dict(config_dict)


class MbagAlphaZero(AlphaZero):
    def __init__(self, config: MbagAlphaZeroConfig, *args, **kwargs):
        del config.ranked_rewards

        super().__init__(config, *args, **kwargs)

        if not config.use_replay_buffer:
            self.local_replay_buffer = None

    @classmethod
    def get_default_config(cls):
        return MbagAlphaZeroConfig()

    def get_default_policy_class(self, config):
        return MbagAlphaZeroPolicy

    def training_step(self) -> ResultDict:
        assert self.workers is not None

        # Sample n MultiAgentBatches from n workers.
        with self._timers[SAMPLE_TIMER]:
            new_sample_batches = synchronous_parallel_sample(
                worker_set=self.workers,
                concat=False,
                max_env_steps=self.config["sample_batch_size"],
            )

        if isinstance(new_sample_batches, list):
            new_sample_batch = concat_samples(new_sample_batches)
        else:
            new_sample_batch = new_sample_batches

        # Update sampling step counters.
        self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
        # Store new samples in the replay buffer.
        if self.local_replay_buffer is not None:
            self.local_replay_buffer.add(new_sample_batch)

        if self.local_replay_buffer is not None:
            cur_ts = self._counters[
                NUM_AGENT_STEPS_SAMPLED
                if self.config.count_steps_by == "agent_steps"
                else NUM_ENV_STEPS_SAMPLED
            ]

            if cur_ts > self.config.num_steps_sampled_before_learning_starts:
                train_batch = self.local_replay_buffer.sample(
                    self.config.train_batch_size
                )
            else:
                train_batch = None
        else:
            train_batch = new_sample_batch

        # Learn on the training batch.
        # Use simple optimizer (only for multi-agent or tf-eager; all other
        # cases should use the multi-GPU optimizer, even if only using 1 GPU)
        train_results = {}
        if train_batch is not None:
            if self.config.get("simple_optimizer") is True:
                train_results = train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)

        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results


register_trainable("MbagAlphaZero", MbagAlphaZero)
