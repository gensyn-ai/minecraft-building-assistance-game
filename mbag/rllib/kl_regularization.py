from typing import List, Mapping

import numpy as np
import torch
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation import MultiAgentBatch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy import TorchPolicy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.sgd import minibatches
from ray.rllib.utils.typing import PolicyID, SampleBatchType

ANCHOR_POLICY_ACTION_DIST_INPUTS = "anchor_policy_action_dist_inputs"


class KLRegularizationMixin(Algorithm):
    """
    Mixin for Algorithm subclasses that helps implement KL regularization.
    """

    def _add_anchor_policy_action_dist_inputs_to_sample_batch(
        self, batch: SampleBatchType
    ) -> MultiAgentBatch:
        assert isinstance(batch, MultiAgentBatch)
        assert self.workers is not None

        anchor_policy_mapping: Mapping[PolicyID, PolicyID] = self.config[
            "anchor_policy_mapping"
        ]
        for policy_id, policy_batch in batch.policy_batches.items():
            anchor_policy_id = anchor_policy_mapping.get(policy_id)
            if anchor_policy_id is None:
                continue

            policy = self.workers.local_worker().get_policy(policy_id)
            assert isinstance(policy, (TorchPolicy, TorchPolicyV2))
            assert isinstance(policy.model, TorchModelV2)

            minibatch_outputs: List[np.ndarray] = []

            for minibatch in minibatches(
                policy_batch,
                self.config["sgd_minibatch_size"],
                shuffle=False,
            ):
                minibatch.decompress_if_needed()

                if not minibatch.zero_padded:
                    pad_batch_to_sequences_of_same_size(
                        batch=minibatch,
                        max_seq_len=policy.max_seq_len,
                        shuffle=False,
                        batch_divisibility_req=policy.batch_divisibility_req,
                        view_requirements=policy.view_requirements,
                    )

                minibatch.set_training(False)
                policy._lazy_tensor_dict(minibatch, device=policy.devices[0])
                with torch.no_grad():
                    model_out, state = policy.model(minibatch)
                    minibatch_outputs.append(model_out.detach().cpu().numpy())

            policy_batch[ANCHOR_POLICY_ACTION_DIST_INPUTS] = np.concatenate(
                minibatch_outputs, axis=0
            )

        return batch
