from typing import Optional, cast

import numpy as np
from ray.rllib.evaluation import SampleBatch

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import MbagConfigDict
from mbag.environment.types import CURRENT_BLOCKS, GOAL_BLOCKS, MbagObs


def randomly_permute_block_types(
    batch: SampleBatch,
    *,
    flat_actions=False,
    env_config: Optional[MbagConfigDict] = None,
) -> SampleBatch:
    new_batch = batch.copy()

    if flat_actions:
        if env_config is None:
            raise ValueError("env_config must be provided if flat_actions is True")
        action_mapping = MbagActionDistribution.get_action_mapping(env_config)
        old_action_block_ids = action_mapping[batch[SampleBatch.ACTIONS]][:, 2]
    else:
        old_action_block_ids = batch[SampleBatch.ACTIONS][2]
    new_action_block_ids = np.empty_like(old_action_block_ids)

    placeable_block_ids = np.array(list(MinecraftBlocks.PLACEABLE_BLOCK_IDS))

    seq_begin = 0
    for seq_len in batch[SampleBatch.SEQ_LENS]:
        seq_end = seq_begin + seq_len

        block_map = np.arange(MinecraftBlocks.NUM_BLOCKS)
        block_map[placeable_block_ids] = np.random.permutation(placeable_block_ids)

        new_world_obs, new_inventory_obs, _ = cast(MbagObs, new_batch[SampleBatch.OBS])
        old_world_obs, old_inventory_obs, _ = cast(MbagObs, batch[SampleBatch.OBS])
        assert new_world_obs is not old_world_obs
        new_world_obs[seq_begin:seq_end, CURRENT_BLOCKS] = block_map[
            old_world_obs[seq_begin:seq_end, CURRENT_BLOCKS]
        ]
        new_world_obs[seq_begin:seq_end, GOAL_BLOCKS] = block_map[
            old_world_obs[seq_begin:seq_end, GOAL_BLOCKS]
        ]

        new_inventory_obs[seq_begin:seq_end] = old_inventory_obs[
            seq_begin:seq_end, :, block_map
        ]

        # Permute actions.
        new_action_block_ids[seq_begin:seq_end] = block_map[
            old_action_block_ids[seq_begin:seq_end]
        ]

        seq_begin = seq_end

    assert seq_end == len(batch)

    if flat_actions:
        if env_config is None:
            raise ValueError("env_config must be provided if flat_actions is True")
        width, height, depth = env_config["world_size"]
        new_batch[SampleBatch.ACTIONS] += (
            (new_action_block_ids - old_action_block_ids) * width * height * depth
        )
    else:
        new_batch[SampleBatch.ACTIONS][2][:] = new_action_block_ids

    return new_batch
