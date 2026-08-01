from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.distributions import Distribution


class WorldModel(ABC):
    @abstractmethod
    def train_step(self, s_batch: torch.Tensor, a_batch: torch.Tensor) -> float: ...

    @abstractmethod
    def predictNextState(self, s: torch.Tensor, a: torch.Tensor) -> Distribution: ...

    @abstractmethod
    def evaluatePredictions(self, s: torch.Tensor, a: torch.Tensor, s_1: torch.Tensor) -> float: ...

    @abstractmethod
    def save_model(self, filepath: str | Path) -> None: ...

    @abstractmethod
    def load_model(self, filepath: str | Path) -> None: ...


class CausalModel(WorldModel):
    @abstractmethod
    def update_mask(self, s_batch: torch.Tensor, a_batch: torch.Tensor) -> None: ...

    @abstractmethod
    def get_causal_graph(self) -> torch.Tensor: ...

    @abstractmethod
    def get_binary_graph(self, threshold: float | None = None) -> torch.Tensor: ...
