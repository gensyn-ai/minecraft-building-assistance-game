import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union, cast

from ray.rllib.policy.torch_policy import TorchPolicy, EntropyCoeffSchedule
from ray.rllib.contrib.alpha_zero.core.alpha_zero_policy import AlphaZeroPolicy
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.contrib.alpha_zero.core.mcts import MCTS, Node, RootParentNode
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models import ModelCatalog, ModelV2, ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import _global_registry, ENV_CREATOR, register_trainable
from ray.rllib.evaluation import SampleBatch
from ray.rllib.utils.typing import TensorType, AgentID
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.models.modelv2 import restore_original_dimensions

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.types import MbagActionTuple, MbagInfoDict, MbagAction
from .rllib_env import unwrap_mbag_env
from .planning import MbagEnvModel
from .torch_models import RewardPredictorMixin


MCTS_POLICIES = "mcts_policies"
ALL_REWARDS = "all_rewards"
OTHER_AGENT_REWARDS = "other_agent_rewards"


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
    children: Dict[int, "MbagMCTSNode"]
    action_mapping: np.ndarray
    parent: Union["MbagMCTSNode", MbagRootParentNode]
    own_reward: np.ndarray
    other_reward: float

    def __init__(
        self,
        action,
        obs,
        done,
        info: Optional[MbagInfoDict],
        reward,
        state,
        mcts,
        parent=None,
    ):
        super().__init__(action, obs, done, reward, state, mcts, parent)

        self.info = info

        self.min_value = np.inf
        self.max_value = -np.inf
        self.action_mapping = self.parent.action_mapping

        self.action_type_dirichlet_noise = None
        self.dirichlet_noise = None

    def child_Q(self):  # noqa: N802
        Q = self.child_total_value / self.child_number_visits  # noqa: N806
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
        Q = total_value / number_visits  # noqa: N806
        V = (  # noqa: N806
            self.total_value / self.number_visits if self.number_visits > 0 else 0
        )
        Q[number_visits == 0] = V
        Q = (Q - self.min_value) / max(  # noqa: N806
            self.max_value - self.min_value, 0.01
        )
        return Q

    def action_type_U(self):  # noqa: N802
        number_visits = np.bincount(
            self.action_mapping[:, 0], weights=self.child_number_visits
        )
        priors = np.bincount(self.action_mapping[:, 0], weights=self.child_priors)
        return np.sqrt(self.number_visits) * priors / (1 + number_visits)

    @property
    def valid_action_types(self) -> np.ndarray:
        return (
            np.bincount(
                self.action_mapping[:, 0], weights=self.valid_actions.astype(np.int32)
            )
            > 0
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

    def expand(self, child_priors, add_dirichlet_noise=False) -> None:
        super().expand(child_priors)

        self.child_priors[~self.valid_actions] = 0
        self.child_priors /= self.child_priors.sum()

        if add_dirichlet_noise:
            num_action_types = self.action_mapping[-1, 0] + 1
            action_type_dirichlet_noise = np.random.dirichlet(
                np.full(num_action_types, 0.25)
            )

            for action_type in range(num_action_types):
                type_mask = (
                    self.action_mapping[:, 0] == action_type
                ) & self.valid_actions
                if not np.any(type_mask):
                    break

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

    def get_child(self, action) -> "MbagMCTSNode":
        if action not in self.children:
            self.env.set_state(self.state)
            obs, reward, done, info = self.env.step(action)
            next_state = self.env.get_state()
            self.children[action] = MbagMCTSNode(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                info=info,
                obs=obs,
                mcts=self.mcts,
            )
        return self.children[action]

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
    def __init__(self, model, mcts_param, gamma: float, use_critic=True):
        super().__init__(model, mcts_param)
        self.gamma = gamma  # Discount factor.
        self.use_critic = use_critic

    def compute_action(self, node: MbagMCTSNode):
        for _ in range(self.num_sims):
            leaf: MbagMCTSNode = node.select()
            if leaf.done:
                value = 0
            else:
                self.model.eval()
                child_priors, value = self.model.compute_priors_and_value(leaf.obs)
                if not self.use_critic:
                    value = 0

                if isinstance(self.model, RewardPredictorMixin):
                    own_reward, other_reward = self.model.predict_reward()
                    leaf.own_reward = MbagActionDistribution.to_flat(
                        node.env.config, convert_to_numpy(own_reward), reduction=np.mean
                    )[0]
                    leaf.other_reward = float(other_reward)
    
                leaf.expand(
                    child_priors,
                    add_dirichlet_noise=self.add_dirichlet_noise and leaf == node,
                )

            if isinstance(self.model, RewardPredictorMixin):
                if (
                    leaf.action is not None
                    and leaf.info is not None
                    and isinstance(leaf.parent, MbagMCTSNode)
                ):
                    action = leaf.action
                    if leaf.info["action_type"] == MbagAction.NOOP:
                        action = 0
                    leaf.reward = (
                        leaf.parent.own_reward[action] + leaf.parent.other_reward
                    )
                else:
                    leaf.reward = 0

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
            logger.debug(
                [
                    (
                        node.get_mbag_action(action),
                        child.reward,
                        node.child_Q()[action],
                        node.child_U()[action],
                        child.number_visits,
                    )
                    for action, child in node.children.items()
                ]
            )

            child = node.children[action]
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
                current = current.children[plan_action]
                plan.append((node.get_mbag_action(plan_action), current.reward))
            logger.debug(plan)

        return tree_policy, action, node.children[action]


class MbagAlphaZeroPolicy(AlphaZeroPolicy, EntropyCoeffSchedule):
    mcts: MbagMCTS
    env: MbagEnvModel

    def __init__(self, obs_space, action_space, config):
        model = ModelCatalog.get_model_v2(
            obs_space, action_space, action_space.n, config["model"], "torch"
        )

        def env_creator():
            env_creator = _global_registry.get(ENV_CREATOR, config["env"])
            env = env_creator(config["env_config"])
            return MbagEnvModel(env, config["env_config"])

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

        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )

    def compute_actions_from_input_dict(
        self, input_dict, explore=None, timestep=None, episodes=None, **kwargs
    ):
        player_index = int(input_dict[SampleBatch.AGENT_INDEX])
        self.env.set_player_index(player_index)

        with torch.no_grad():
            actions = []
            for i, episode in enumerate(episodes):
                env_state = episode.user_data["state"]
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
                    mcts=self.mcts,
                )

                # run monte carlo simulations to compute the actions
                # and record the tree
                mcts_policy, action, tree_node = self.mcts.compute_action(tree_node)
                # record action
                actions.append(action)

                # store mcts policies vectors and current tree root node
                if episode.length == 0:
                    episode.user_data[MCTS_POLICIES] = [mcts_policy]
                else:
                    episode.user_data[MCTS_POLICIES].append(mcts_policy)

            return (
                np.array(actions),
                [],
                self.extra_action_out(
                    input_dict, kwargs.get("state_batches", []), self.model, None
                ),
            )

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple[Type[TorchPolicy], SampleBatch]]
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

        # Add all rewards to sample batch for the purposes of training the reward
        # predictor.
        obs = restore_original_dimensions(
            sample_batch[SampleBatch.OBS], self.observation_space, tensorlib=np
        )
        sample_batch[ALL_REWARDS] = self.env.get_all_rewards(obs)

        sample_batch[OTHER_AGENT_REWARDS] = np.zeros(len(sample_batch))
        if other_agent_batches is not None:
            for agent_id, (_, other_agent_batch) in other_agent_batches.items():
                infos: Sequence[MbagInfoDict] = other_agent_batch[SampleBatch.INFOS]
                sample_batch[OTHER_AGENT_REWARDS] += np.array(
                    np.array([info["own_reward"] for info in infos])
                )

        return sample_batch

    def learn_on_batch(self, postprocessed_batch):
        return TorchPolicy.learn_on_batch(self, postprocessed_batch)

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        assert isinstance(model, TorchModelV2)

        # Forward pass in model.
        model_out = model(train_batch, None, [1])
        logits, _ = model_out
        values = model.value_function()
        logits, values = torch.squeeze(logits), torch.squeeze(values)
        dist: TorchCategorical = dist_class(logits)

        # Compute actor and critic losses.
        dist.logp
        policy_loss = torch.mean(
            -torch.sum(train_batch[MCTS_POLICIES] * dist.dist.logits, dim=-1)
        )
        value_loss = torch.mean(
            (values - train_batch[Postprocessing.VALUE_TARGETS]) ** 2
        )

        entropy = dist.entropy().mean()

        # Compute total loss.
        total_loss = (
            policy_loss
            + self.config["vf_loss_coeff"] * value_loss
            - self.entropy_coeff * entropy
        )

        if isinstance(model, RewardPredictorMixin):
            # Compute reward predictor losses.
            own_rewards, other_rewards = model.predict_reward()
            own_reward_loss = torch.mean((own_rewards - train_batch[ALL_REWARDS]) ** 2)
            other_reward_loss = torch.mean(
                (other_rewards - train_batch[OTHER_AGENT_REWARDS]) ** 2
            )
            reward_predictor_loss = own_reward_loss + other_reward_loss
            total_loss = (
                total_loss
                + self.config["reward_predictor_loss_coeff"] * reward_predictor_loss
            )

            model.tower_stats["own_reward_loss"] = own_reward_loss
            model.tower_stats["own_reward_explained_var"] = explained_variance(
                train_batch[ALL_REWARDS].flatten(),
                own_rewards.flatten(),
            )
            model.tower_stats["other_reward_loss"] = other_reward_loss
            model.tower_stats["other_reward_explained_var"] = explained_variance(
                train_batch[OTHER_AGENT_REWARDS],
                other_rewards.flatten(),
            )
            model.tower_stats["reward_predictor_loss"] = reward_predictor_loss

        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["policy_loss"] = policy_loss
        model.tower_stats["vf_loss"] = value_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], values
        )
        model.tower_stats["entropy"] = entropy

        return total_loss

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        grad_info: Dict[str, TensorType] = {
            "entropy_coeff": self.entropy_coeff,
        }
        for metric in [
            "total_loss",
            "policy_loss",
            "vf_loss",
            "vf_explained_var",
            "entropy",
            "own_reward_loss",
            "own_reward_explained_var",
            "other_reward_loss",
            "other_reward_explained_var",
            "reward_predictor_loss",
        ]:
            if self.get_tower_stats(metric):
                grad_info[metric] = torch.mean(
                    torch.stack(self.get_tower_stats(metric))
                ).item()
        return grad_info

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"]
            )


class MbagAlphaZeroTrainer(AlphaZeroTrainer):
    @classmethod
    def get_default_config(cls):
        config = {
            **AlphaZeroTrainer.get_default_config(),
            "vf_loss_coeff": 1.0,
            "reward_predictor_loss_coeff": 1.0,
            "entropy_coeff": 0,
            "entropy_coeff_schedule": 0,
            "use_critic": True,
        }
        del config["vf_share_layers"]
        return config

    def get_default_policy_class(self, config):
        return MbagAlphaZeroPolicy


register_trainable("MbagAlphaZero", MbagAlphaZeroTrainer)
