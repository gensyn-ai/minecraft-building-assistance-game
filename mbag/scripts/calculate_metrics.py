import json
import logging
import os
import pickle
from typing import List

import tqdm
from sacred import Experiment

from mbag.evaluation.episode import MbagEpisode
from mbag.evaluation.metrics import (
    MbagEpisodeMetrics,
    calculate_mean_metrics,
    calculate_metrics,
)

ex = Experiment()


@ex.config
def sacred_config():
    evaluate_dir = ""
    out_fname = os.path.join(evaluate_dir, "metrics.json")  # noqa: F401


@ex.automain
def main(  # noqa: C901
    evaluate_dir: str,
    out_fname: str,
    _log: logging.Logger,
):
    episodes_fname = os.path.join(evaluate_dir, "episodes.pickle")
    if not os.path.exists(episodes_fname):
        episodes_fname = os.path.join(evaluate_dir, "episode_info.pickle")
    _log.info(f"loading episodes from {episodes_fname}...")
    with open(episodes_fname, "rb") as episodes_file:
        episodes: List[MbagEpisode] = pickle.load(episodes_file)

    episode_metrics: List[MbagEpisodeMetrics] = []
    for episode in tqdm.tqdm(episodes):
        episode_metrics.append(calculate_metrics(episode))

    results = {
        "mean_metrics": calculate_mean_metrics(episode_metrics),
        "episode_metrics": episode_metrics,
    }
    with open(out_fname, "w") as out_file:
        json.dump(results, out_file)

    return results
