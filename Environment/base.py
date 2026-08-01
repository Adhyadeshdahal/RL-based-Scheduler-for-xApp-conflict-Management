from typing import Callable, List, Tuple

import gymnasium as gym
import numpy as np


def set_param_default(threshold, value, params=None):
    low, high = threshold
    return np.clip(value, low, high)


class XApp:
    def __init__(self, threshold, utility_fn, name, params, direction, mean_std):
        self.thresholds = threshold
        self.threshold = threshold
        self.compute_utility = utility_fn
        self.name = name
        self.params = params
        self.direction = direction
        self.mean, self.std = mean_std


class Param:
    def __init__(self, threshold: Tuple[float, float], id: int, set_param_fn=set_param_default):
        self.threshold = threshold
        self.id = id
        self.value = np.random.uniform(*threshold)
        self.set_param_fn = set_param_fn

    def get_param(self):
        return self.value

    def set_param(self, value, params=None):
        self.value = self.set_param_fn(self.threshold, value, params)

    def get_threshold(self):
        return self.threshold


class KPI:
    def __init__(
        self,
        name: str,
        kpi_threshold: float,
        direction: int,
        mean: float,
        std: float,
        updatefn: Callable,
    ):
        self.name = name
        self.updatefn = updatefn
        self.value = 0.0
        self.kpi_threshold = kpi_threshold
        self.direction = direction
        self.mean = mean
        self.std = std

    def update_kpi(self, prev_params: List[float], prev_kpis: List[float]):
        self.value = self.updatefn(prev_params, prev_kpis)
        return self.value

    def compute_utility_value(self):
        return (self.value - self.mean) / self.std

    def compute_utility_threshold(self):
        return (self.kpi_threshold - self.mean) / self.std


class BaseORANEnv(gym.Env):
    def __init__(
        self,
        param_ranges,
        mean_std_kpis,
        kpi_thresholds,
        kpi_names,
        directions,
        update_fns,
        xapp_param_indices,
        xapp_kpi_indices,
        kpi_to_xapp,
        adjacency_edges,
        num_bins,
        max_steps,
    ):
        super().__init__()
        self.paramThresholds = param_ranges
        self.mean_std_kpis = mean_std_kpis
        self.kpi_thresholds = kpi_thresholds
        self.params = [Param(threshold, index) for index, threshold in enumerate(param_ranges)]
        self.kpis = [
            KPI(name, threshold, direction, mean, std, update_fn)
            for name, threshold, direction, (mean, std), update_fn in zip(
                kpi_names, kpi_thresholds, directions, mean_std_kpis, update_fns
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
            for index, (params, indices) in enumerate(zip(xapp_param_indices, xapp_kpi_indices))
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
        self.action_space = [self.num_params - 1, num_bins - 1] + self.bin_length_per_param
        self.max_steps = max_steps
        self.cur_step = 0
        self.prev_params = None
        self.prev_kpis = None
        self.reset()

    @staticmethod
    def _mean_for_indices(values, indices):
        if len(indices) == 1:
            return values[indices[0]]
        first_value = values[indices[0]]
        if isinstance(first_value, tuple):
            return tuple(
                sum(values[index][part] for index in indices) / len(indices) for part in range(2)
            )
        return sum(values[index] for index in indices) / len(indices)

    def _utility_for_indices(self, indices):
        mean, std = self._mean_for_indices(self.mean_std_kpis, indices)

        def utility(kpis):
            value = sum(kpis[index] for index in indices) / len(indices)
            return (value - mean) / std

        return utility

    def reset(self):
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

    def reward(self, new_kpis, next_kpis):
        return sum(next_kpis)

    def step(self, action: Tuple[int, int, int]):
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

    def get_state(self):
        state = {}
        for index, value in enumerate(self.prev_params):
            low, high = self.params[index].get_threshold()
            normalized = (value - low) / (high - low) if high != low else 1.0
            state[f"param{index}"] = np.array([normalized], dtype=np.float32)
        for kpi in self.kpis:
            value = kpi.compute_utility_value() + np.random.normal(0, 0.01)
            state[kpi.name] = np.array([value], dtype=np.float32)
        return state

    def get_state_dim(self):
        return self.num_params + self.num_kpis

    def get_action_dim(self):
        return self.action_dim

    def action_to_param(self, action):
        param_id, bin_id, index = action
        low, high = self.paramThresholds[param_id]
        value = low + (high - low) * (bin_id / (self.num_bins - 1)) + index
        return param_id, np.clip(value, low, high)

    def get_utility_fns(self):
        utility_fns = []
        for xapp in self.xapps:

            def utility(params, xapp=xapp):
                kpis = [0.0] * self.num_kpis
                for index, kpi in enumerate(self.kpis):
                    kpis[index] = kpi.updatefn(params, kpis)
                return xapp.compute_utility(kpis)

            utility_fns.append(utility)
        return utility_fns

    def get_thresholds_stds(self):
        return [xapp.threshold for xapp in self.xapps], [
            (xapp.mean, xapp.std) for xapp in self.xapps
        ]

    def get_kpi_to_xapp_mapping(self):
        return self._kpi_to_xapp
