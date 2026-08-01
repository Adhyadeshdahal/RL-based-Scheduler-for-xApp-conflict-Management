from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.distributions import Distribution


class WorldModel(ABC):
    kpi_start: int
    node_names: list[str]

    @abstractmethod
    def train_step(self, s_batch: torch.Tensor, a_batch: torch.Tensor) -> float: ...

    @abstractmethod
    def predict_next_state(self, s: torch.Tensor, a: torch.Tensor) -> Distribution: ...

    @abstractmethod
    def evaluate_predictions(
        self, s: torch.Tensor, a: torch.Tensor, s_1: torch.Tensor
    ) -> float: ...

    @abstractmethod
    def save_model(self, filepath: str | Path) -> None: ...

    @abstractmethod
    def load_model(self, filepath: str | Path) -> None: ...


class CausalModel(WorldModel):
    cmi_threshold: float

    @abstractmethod
    def update_mask(self, s_batch: torch.Tensor, a_batch: torch.Tensor) -> None: ...

    @abstractmethod
    def get_causal_graph(self) -> torch.Tensor: ...

    @abstractmethod
    def get_binary_graph(self, threshold: float | None = None) -> torch.Tensor: ...
