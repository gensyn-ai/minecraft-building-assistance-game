import logging
import os
import glob
from ray.rllib.offline.json_reader import _from_json
from ray.rllib.offline.json_writer import JsonWriter, logger as json_writer_logger
from ray.rllib.policy.sample_batch import MultiAgentBatch
from sacred.experiment import Experiment


ex = Experiment("split_rollouts")


@ex.config
def sacred_config():
    in_dir = ""
    out_dir = os.path.join(in_dir, "split")  # noqa: F841
    rollout_fragment_length = 100  # noqa: F841
    max_file_size = 1  # noqa: F841


@ex.automain
def main(
    in_dir: str,
    out_dir: str,
    rollout_fragment_length: int,
    max_file_size: int,
    _log,
):
    json_writer_logger.setLevel(logging.WARN)

    writer = JsonWriter(out_dir, max_file_size=max_file_size)
    for rollout_fname in sorted(glob.glob(os.path.join(in_dir, "*.json"))):
        _log.info(f"reading {rollout_fname}...")
        with open(rollout_fname, "r") as rollout_file:
            for rollout_line in rollout_file:
                sample_batch = _from_json(rollout_line.strip())
                assert isinstance(sample_batch, MultiAgentBatch)

                slice_starts = range(0, sample_batch.count, rollout_fragment_length)
                _log.info(f"splitting into {len(slice_starts)} slices...")
                for slice_start in slice_starts:
                    slice_end = slice_start + rollout_fragment_length
                    if slice_end > sample_batch.count:
                        slice_end = sample_batch.count
                    slice_batch = MultiAgentBatch(
                        {
                            policy_id: policy_batch.slice(slice_start, slice_end)
                            for policy_id, policy_batch in sample_batch.policy_batches.items()
                        },
                        slice_end - slice_start,
                    )
                    writer.write(slice_batch)
