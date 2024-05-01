import json
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np

from mbag.environment.actions import MbagAction
from mbag.environment.config import MbagConfigDict
from mbag.environment.types import MbagInfoDict, MbagObs


@dataclass
class MbagEpisode:
    env_config: MbagConfigDict
    reward_history: List[float]
    cumulative_reward: float
    length: int
    obs_history: List[List[MbagObs]]
    last_obs: List[MbagObs]
    info_history: List[List[MbagInfoDict]]
    last_infos: List[MbagInfoDict]

    def to_json(self) -> dict:
        return {
            "cumulative_reward": self.cumulative_reward,
            "length": self.length,
            "last_infos": self.last_infos,
        }


class EpisodeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, MbagAction):
            return obj.to_tuple()
        return super().default(obj)
