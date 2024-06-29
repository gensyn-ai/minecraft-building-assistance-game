import logging
from typing import Dict, List, Sequence, Union, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from gymnasium import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn

from mbag.rllib.torch_models import MbagTorchModel

logger = logging.getLogger(__name__)


class MixtureModel(TorchModelV2, nn.Module):
    """
    Represents a mixture of different models. The first element of the state is the
    current distribution over the mixture; by default, it puts all the mass on a
    randomly chosen model at the beginning of each episode.

    Alternatively, the first element of the initial state can be set to some other
    distribution (e.g., a uniform distribution), in which case the model will do
    inference over the mixture distribution according the ACTIONS passed in the
    input dictionary.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        model_configs: List[ModelConfigDict] = kwargs["model_configs"]

        self.components = cast(
            Sequence[TorchModelV2],
            nn.ModuleList(
                [
                    cast(
                        nn.Module,
                        ModelCatalog.get_model_v2(
                            obs_space,
                            action_space,
                            num_outputs,
                            model_config,
                            framework="torch",
                            name=f"{name}_component_{model_index}",
                        ),
                    )
                    for model_index, model_config in enumerate(model_configs)
                ]
            ),
        )
        self.component_state_lens: List[int] = [
            len(model.get_initial_state()) for model in self.components
        ]

    def get_initial_state(self) -> List[Union[np.ndarray, torch.Tensor]]:
        state: List[Union[np.ndarray, torch.Tensor]] = [
            torch.zeros((len(self.components),))
        ]
        for model in self.components:
            state.extend(model.get_initial_state())
        return state

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        batch_size = obs.size()[0]
        device = obs.device
        num_components = len(self.components)
        seq_lens = seq_lens.to(device)

        start_mixture_logprobs, *component_states = state  # shape (B, N)
        no_dist_mask = torch.all(start_mixture_logprobs == 0, dim=1)
        start_mixture_logprobs[no_dist_mask, :] = (
            F.one_hot(
                torch.randint(
                    num_components, (int(no_dist_mask.sum().item()),), device=device
                ),
                num_classes=num_components,
            )
            .float()
            .log()
        )

        components_out = torch.zeros(  # shape (N, BT, A)
            (num_components, batch_size, self.num_outputs),
            device=device,
        )
        component_states_out: List[torch.Tensor] = []
        component_state_start_index = 0
        for component_index, component in enumerate(self.components):
            support_mask = (  # shape (B,)
                start_mixture_logprobs[:, component_index] > -torch.inf
            )
            if torch.any(support_mask):
                full_batch_support_mask = torch.empty(  # shape (BT,)
                    batch_size, device=device, dtype=torch.bool
                )
                support_mask_with_time_dim = add_time_dimension(  # shape (B, T)
                    full_batch_support_mask,
                    seq_lens=seq_lens,
                    time_major=False,
                    framework="torch",
                )
                support_mask_with_time_dim[:] = support_mask[:, None]

                component_state_len = self.component_state_lens[component_index]
                component_state_end_index = (
                    component_state_start_index + component_state_len
                )
                component_state_in = component_states[
                    component_state_start_index:component_state_end_index
                ]
                component_state_start_index += component_state_len

                component_input_dict: Dict[str, Union[np.ndarray, torch.Tensor]] = {
                    key: val[full_batch_support_mask]
                    for key, val in input_dict.items()
                    if isinstance(val, torch.Tensor)
                }
                component_input_dict["obs"] = obs[full_batch_support_mask]

                component_out, component_state_out = component(
                    component_input_dict,
                    [state_piece[support_mask] for state_piece in component_state_in],
                    seq_lens[support_mask],
                )
                components_out[component_index][full_batch_support_mask] = component_out

                full_component_state_out: List[torch.Tensor] = []
                for state_piece_in, state_piece_out in zip(
                    component_state_in, component_state_out
                ):
                    full_state_piece = state_piece_in.clone()
                    full_state_piece[support_mask] = state_piece_out
                    full_component_state_out.append(full_state_piece)
                component_states_out.extend(full_component_state_out)

        component_logprobs = torch.log_softmax(
            components_out, dim=2
        )  # shape (N, BT, A)
        component_logprobs_with_time_dim = add_time_dimension(  # shape (B, T, N, A)
            component_logprobs.permute(1, 0, 2),
            seq_lens=seq_lens,
            time_major=False,
            framework="torch",
        )

        full_batch_mixture_logprobs = torch.empty(  # shape (BT, N)
            batch_size,
            num_components,
            device=device,
            dtype=start_mixture_logprobs.dtype,
        )
        full_batch_mixture_logprobs_with_time_dim = (
            add_time_dimension(  # shape (B, T, N)
                full_batch_mixture_logprobs,
                seq_lens=seq_lens,
                time_major=False,
                framework="torch",
            )
        )
        full_batch_mixture_logprobs_with_time_dim[:] = start_mixture_logprobs[
            :, None, :
        ]

        if SampleBatch.ACTIONS in input_dict:
            actions_with_time_dim = add_time_dimension(  # shape (B, T)
                input_dict[SampleBatch.ACTIONS],
                seq_lens=seq_lens,
                time_major=False,
                framework="torch",
            )
            action_logprobs = torch.gather(  # shape (B, T, N)
                component_logprobs_with_time_dim,
                dim=3,
                index=actions_with_time_dim[:, :, None, None].expand(
                    -1, -1, num_components, -1
                ),
            ).squeeze(3)
            action_logprobs_cumsum = torch.cumsum(
                action_logprobs, dim=1
            )  # shape (B, T, N)
            full_batch_mixture_logprobs_with_time_dim[:, 1:] += action_logprobs_cumsum[
                :, :-1
            ]
            full_batch_mixture_logprobs_with_time_dim[:] = torch.log_softmax(
                full_batch_mixture_logprobs_with_time_dim, dim=2
            )
            final_mixture_logprobs = torch.log_softmax(  # shape (B, N)
                start_mixture_logprobs + action_logprobs_cumsum[:, -1], dim=1
            )
        else:
            final_mixture_logprobs = start_mixture_logprobs.clone()

        # log sum_i p_i * exp^(log pi_i) = log sum_i exp(log p_i + log pi_i)
        component_logprobs += full_batch_mixture_logprobs.permute(1, 0)[:, :, None]
        model_out = torch.logsumexp(component_logprobs, dim=0)

        assert ~torch.any(torch.isnan(model_out))

        self._vf = torch.zeros((batch_size,), device=device)

        return model_out, [final_mixture_logprobs, *component_states_out]

    @property
    def env_config(self):
        assert isinstance(self.components[0], MbagTorchModel)
        return self.components[0].env_config

    def value_function(self):
        logger.warn("value function is not implemented for mixture models")
        return self._vf


ModelCatalog.register_custom_model("mbag_mixture", MixtureModel)
