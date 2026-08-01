from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np


def set_param_default(
    threshold: tuple[float, float], value: float, params: Sequence[Param] | None = None
) -> float:
    low, high = threshold
    return float(np.clip(value, low, high))


class XApp:
    def __init__(
        self,
        threshold: float,
        utility_fn: Callable[[list[float]], float],
        name: str,
        params: tuple[Param, ...],
        direction: int,
        mean_std: tuple[float, float],
    ) -> None:
        self.thresholds = threshold
        self.threshold = threshold
        self.compute_utility = utility_fn
        self.name = name
        self.params = params
        self.direction = direction
        self.mean, self.std = mean_std


class Param:
    def __init__(
        self,
        threshold: tuple[float, float],
        id: int,
        set_param_fn: Callable[..., float] = set_param_default,
    ) -> None:
        self.threshold = threshold
        self.id = id
        self.value = np.random.uniform(*threshold)
        self.set_param_fn = set_param_fn

    def get_param(self) -> float:
        return self.value

    def set_param(self, value: float, params: Sequence[Param] | None = None) -> None:
        self.value = self.set_param_fn(self.threshold, value, params)

    def get_threshold(self) -> tuple[float, float]:
        return self.threshold


class KPI:
    def __init__(
        self,
        name: str,
        kpi_threshold: float,
        direction: int,
        mean: float,
        std: float,
        updatefn: Callable[[list[float], list[float]], float],
    ) -> None:
        self.name = name
        self.updatefn = updatefn
        self.value = 0.0
        self.kpi_threshold = kpi_threshold
        self.direction = direction
        self.mean = mean
        self.std = std

    def update_kpi(self, prev_params: list[float], prev_kpis: list[float]) -> float:
        self.value = self.updatefn(prev_params, prev_kpis)
        return self.value

    def compute_utility_value(self) -> float:
        return (self.value - self.mean) / self.std

    def compute_utility_threshold(self) -> float:
        return (self.kpi_threshold - self.mean) / self.std


class BaseORANEnv(gym.Env[Any, Any]):
    def __init__(
        self,
        param_ranges: Sequence[tuple[float, float]],
        mean_std_kpis: Sequence[tuple[float, float]],
        kpi_thresholds: Sequence[float],
        kpi_names: Sequence[str],
        directions: Sequence[int],
        update_fns: Sequence[Callable[[list[float], list[float]], float]],
        xapp_param_indices: Sequence[tuple[int, ...]],
        xapp_kpi_indices: Sequence[tuple[int, ...]],
        kpi_to_xapp: dict[int, int],
        adjacency_edges: Sequence[tuple[int, int]],
        num_bins: int,
        max_steps: int,
    ) -> None:
        super().__init__()
        self.paramThresholds = param_ranges
        self.mean_std_kpis = mean_std_kpis
        self.kpi_thresholds = kpi_thresholds
        self.params = [Param(threshold, index) for index, threshold in enumerate(param_ranges)]
        self.kpis = [
            KPI(name, threshold, direction, mean, std, update_fn)
            for name, threshold, direction, (mean, std), update_fn in zip(
                kpi_names, kpi_thresholds, directions, mean_std_kpis, update_fns, strict=True
            )
        ]
        self.num_params = len(self.params)
        self.num_kpis = len(self.kpis)
        self.xapps = [
            XApp(
                threshold=self._mean_for_indices(kpi_thresholds, indices),
                utility_fn=self._utility_for_indices(indices),
                name=f"xApp{index}",
                params=tuple(self.params[param] for param in params),
                direction=directions[indices[0]],
                mean_std=self._mean_for_indices(mean_std_kpis, indices),
            )
            for index, (params, indices) in enumerate(
                zip(xapp_param_indices, xapp_kpi_indices, strict=True)
            )
        ]
        self._kpi_to_xapp = {
            self.num_params + kpi_index: xapp_index for kpi_index, xapp_index in kpi_to_xapp.items()
        }
        self.true_adj_matrix = np.zeros(
            (self.num_params + self.num_kpis, self.num_params + self.num_kpis), dtype=np.float32
        )
        for kpi_index, param_index in adjacency_edges:
            self.true_adj_matrix[self.num_params + kpi_index, param_index] = 1
        self.num_bins = num_bins
        self.action_dim = 3
        self.bin_length_per_param = [int((high - low) // num_bins) for low, high in param_ranges]
        self.action_space: Any = [self.num_params - 1, num_bins - 1] + self.bin_length_per_param
        self.max_steps = max_steps
        self.cur_step = 0
        self.prev_params: list[float] = []
        self.prev_kpis: list[float] = []
        self.reset()

    @staticmethod
    def _mean_for_indices(values: Sequence[Any], indices: Sequence[int]) -> Any:
        if len(indices) == 1:
            return values[indices[0]]
        first_value = values[indices[0]]
        if isinstance(first_value, tuple):
            return tuple(
                sum(values[index][part] for index in indices) / len(indices) for part in range(2)
            )
        return sum(values[index] for index in indices) / len(indices)

    def _utility_for_indices(self, indices: Sequence[int]) -> Callable[[list[float]], float]:
        mean, std = cast(tuple[float, float], self._mean_for_indices(self.mean_std_kpis, indices))

        def utility(kpis: list[float]) -> float:
            value = sum(kpis[index] for index in indices) / len(indices)
            return (value - mean) / std

        return utility

    def reset(self) -> dict[str, np.ndarray]:  # ty: ignore[invalid-method-override] -- Legacy Gym 4-tuple API is intentional.
        self.cur_step = 0
        for param in self.params:
            low, high = param.get_threshold()
            value = np.random.uniform(low, high)
            if low == high:
                value = low
            param.set_param(value, self.params)
        self.prev_params = [param.get_param() for param in self.params]
        self.prev_kpis = [0.0 for _ in self.kpis]
        return self.get_state()

    def reward(self, new_kpis: list[KPI], next_kpis: list[float]) -> float:
        return sum(next_kpis)

    def step(  # ty: ignore[invalid-method-override] -- Legacy Gym 4-tuple API is intentional.
        self, action: tuple[int, int, int] | np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, bool]]:
        param_id, bin_id, index = (int(value) for value in action)
        low, high = self.paramThresholds[param_id]
        value = low + (high - low) * (bin_id / (self.num_bins - 1)) + index
        self.params[param_id].set_param(value, self.params)
        new_params = [param.get_param() for param in self.params]
        new_kpis = [kpi.update_kpi(self.prev_params, self.prev_kpis) for kpi in self.kpis]
        reward = self.reward(self.kpis, new_kpis)
        self.prev_params = new_params
        self.prev_kpis = new_kpis
        self.cur_step += 1
        return self.get_state(), reward, self.cur_step >= self.max_steps, {"success": False}

    def get_state(self) -> dict[str, np.ndarray]:
        state = {}
        for index, value in enumerate(self.prev_params):
            low, high = self.params[index].get_threshold()
            normalized = (value - low) / (high - low) if high != low else 1.0
            state[f"param{index}"] = np.array([normalized], dtype=np.float32)
        for kpi in self.kpis:
            value = kpi.compute_utility_value() + np.random.normal(0, 0.01)
            state[kpi.name] = np.array([value], dtype=np.float32)
        return state

    def get_state_dim(self) -> int:
        return self.num_params + self.num_kpis

    def get_action_dim(self) -> int:
        return self.action_dim

    def action_to_param(self, action: tuple[int, int, int]) -> tuple[int, float]:
        param_id, bin_id, index = action
        low, high = self.paramThresholds[param_id]
        value = low + (high - low) * (bin_id / (self.num_bins - 1)) + index
        return param_id, float(np.clip(value, low, high))

    def get_utility_fns(self) -> list[Callable[[list[float]], float]]:
        utility_fns: list[Callable[[list[float]], float]] = []
        for xapp in self.xapps:

            def utility(params: list[float], xapp: XApp = xapp) -> float:
                kpis = [0.0] * self.num_kpis
                for index, kpi in enumerate(self.kpis):
                    kpis[index] = kpi.updatefn(params, kpis)
                return xapp.compute_utility(kpis)

            utility_fns.append(cast(Callable[[list[float]], float], utility))
        return utility_fns

    def get_thresholds_stds(self) -> tuple[list[float], list[tuple[float, float]]]:
        return [xapp.threshold for xapp in self.xapps], [
            (xapp.mean, xapp.std) for xapp in self.xapps
        ]

    def get_kpi_to_xapp_mapping(self) -> dict[int, int]:
        return self._kpi_to_xapp
