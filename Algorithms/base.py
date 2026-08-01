from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class Planner(ABC):
    name: str

    @abstractmethod
    def act(
        self,
        current_state: torch.Tensor,
        conflict_param_index: int,
        xapps_under_conflict: Sequence[object],
        weights_per_xapps: list[float],
        scaling_term: float,
    ) -> list[int]: ...
