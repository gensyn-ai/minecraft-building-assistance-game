import torch
import numpy as np
from typing import Dict, List, Type, Union, cast

from ray.rllib.policy.torch_policy import TorchPolicy, EntropyCoeffSchedule
from ray.rllib.contrib.alpha_zero.core.alpha_zero_policy import AlphaZeroPolicy
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.contrib.alpha_zero.core.mcts import MCTS, Node, RootParentNode
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models import ModelCatalog, ModelV2, ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import _global_registry, ENV_CREATOR, register_trainable
from ray.rllib.evaluation import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.torch_utils import explained_variance

from mbag.agents.action_distributions import MbagActionDistribution
from .rllib_env import unwrap_mbag_env


class DenseRewardRootParentNode(RootParentNode):
    def __init__(self, env):
        super().__init__(env)
        self.action_mapping = MbagActionDistribution.get_action_mapping(
            unwrap_mbag_env(self.env).config
        )


class DenseRewardMCTSNode(Node):
    mcts: "DenseRewardMCTS"
    children: Dict[int, "DenseRewardMCTSNode"]
    action_mapping: np.ndarray
    parent: Union["DenseRewardMCTSNode", DenseRewardRootParentNode]

    def __init__(self, action, obs, done, reward, state, mcts, parent=None):
        super().__init__(action, obs, done, reward, state, mcts, parent)

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

    def get_child(self, action) -> "DenseRewardMCTSNode":
        if action not in self.children:
            self.env.set_state(self.state)
            obs, reward, done, _ = self.env.step(action)
            next_state = self.env.get_state()
            self.children[action] = DenseRewardMCTSNode(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
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
                if isinstance(node, DenseRewardMCTSNode):
                    node.min_value = min(node.min_value, value)
                    node.max_value = max(node.max_value, value)
            if isinstance(current.parent, DenseRewardMCTSNode):
                current = current.parent
            else:
                break


class DenseRewardMCTS(MCTS):
    def __init__(self, model, mcts_param, gamma: float, use_critic=True):
        super().__init__(model, mcts_param)
        self.gamma = gamma  # Discount factor.
        self.use_critic = use_critic

    def compute_action(self, node: DenseRewardMCTSNode):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = 0
            else:
                self.model.eval()
                child_priors, value = self.model.compute_priors_and_value(leaf.obs)
                if not self.use_critic:
                    value = 0

                leaf.expand(
                    child_priors,
                    add_dirichlet_noise=self.add_dirichlet_noise and leaf == node,
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

        return tree_policy, action, node.children[action]


class MbagAlphaZeroPolicy(AlphaZeroPolicy, EntropyCoeffSchedule):
    mcts: DenseRewardMCTS

    def __init__(self, obs_space, action_space, config):
        model = ModelCatalog.get_model_v2(
            obs_space, action_space, action_space.n, config["model"], "torch"
        )
        env_creator = _global_registry.get(ENV_CREATOR, config["env"])

        def _env_creator():
            return env_creator(config["env_config"])

        def mcts_creator():
            return DenseRewardMCTS(model, config["mcts_config"], config["gamma"])

        super().__init__(
            obs_space,
            action_space,
            config,
            model=model,
            loss=None,
            action_distribution_class=TorchCategorical,
            mcts_creator=mcts_creator,
            env_creator=_env_creator,
        )

        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )

    def compute_actions_from_input_dict(
        self, input_dict, explore=None, timestep=None, episodes=None, **kwargs
    ):
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
                tree_node = DenseRewardMCTSNode(
                    state=env_state,
                    obs=obs,
                    reward=0,
                    done=False,
                    action=None,
                    parent=DenseRewardRootParentNode(env=self.env),
                    mcts=self.mcts,
                )

                # run monte carlo simulations to compute the actions
                # and record the tree
                mcts_policy, action, tree_node = self.mcts.compute_action(tree_node)
                # record action
                actions.append(action)

                # store mcts policies vectors and current tree root node
                if episode.length == 0:
                    episode.user_data["mcts_policies"] = [mcts_policy]
                else:
                    episode.user_data["mcts_policies"].append(mcts_policy)

            return (
                np.array(actions),
                [],
                self.extra_action_out(
                    input_dict, kwargs.get("state_batches", []), self.model, None
                ),
            )

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            last_r = 0
            rewards_plus_v = np.concatenate(
                [sample_batch[SampleBatch.REWARDS], np.array([last_r])]
            )
            discounted_returns = discount_cumsum(rewards_plus_v, self.config["gamma"])[
                :-1
            ].astype(np.float32)
            sample_batch["value_label"] = discounted_returns

        # add mcts policies to sample batch
        sample_batch["mcts_policies"] = np.array(episode.user_data["mcts_policies"])[
            sample_batch["t"]
        ]

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

        # forward pass in model
        model_out = model(train_batch, None, [1])
        logits, _ = model_out
        values = model.value_function()
        logits, values = torch.squeeze(logits), torch.squeeze(values)
        dist: TorchCategorical = dist_class(logits)

        # compute actor and critic losses
        policy_loss = torch.mean(
            -torch.sum(
                train_batch["mcts_policies"] * torch.log(dist.dist.probs), dim=-1
            )
        )

        value_loss = torch.mean(torch.pow(values - train_batch["value_label"], 2))

        entropy = dist.entropy().mean()

        # compute total loss
        total_loss = (
            policy_loss
            + self.config["vf_loss_coeff"] * value_loss
            - self.entropy_coeff * entropy
        )

        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["policy_loss"] = policy_loss
        model.tower_stats["value_loss"] = value_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch["value_label"], values
        )
        model.tower_stats["mean_entropy"] = entropy

        return total_loss

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return cast(
            Dict[str, TensorType],
            convert_to_numpy(
                {
                    "total_loss": torch.mean(
                        torch.stack(self.get_tower_stats("total_loss"))
                    ),
                    "policy_loss": torch.mean(
                        torch.stack(self.get_tower_stats("policy_loss"))
                    ),
                    "vf_loss": torch.mean(
                        torch.stack(self.get_tower_stats("value_loss"))
                    ),
                    "vf_explained_var": torch.mean(
                        torch.stack(self.get_tower_stats("vf_explained_var"))
                    ),
                    "entropy": torch.mean(
                        torch.stack(self.get_tower_stats("mean_entropy"))
                    ),
                    "entropy_coeff": self.entropy_coeff,
                }
            ),
        )

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"]
            )


class MbagAlphaZeroTrainer(AlphaZeroTrainer):
    @classmethod
    def get_default_config(cls):
        return {
            **AlphaZeroTrainer.get_default_config(),
            "vf_loss_coeff": 1.0,
            "entropy_coeff": 0,
            "entropy_coeff_schedule": 0,
            "use_critic": True,
        }

    def get_default_policy_class(self, config):
        return MbagAlphaZeroPolicy


register_trainable("MbagAlphaZero", MbagAlphaZeroTrainer)
