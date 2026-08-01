from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def weighted_distance(xapp: Any, utility: float) -> tuple[float, int]:
    mean, std = xapp.mean, xapp.std
    norm_threshold = (xapp.threshold - mean) / std

    if xapp.direction == 0:
        if utility < norm_threshold:
            return norm_threshold - utility, 0
        return 0.0, 1
    if utility > norm_threshold:
        return utility - norm_threshold, 0
    return 0.0, 1


def score_batch(
    next_kpis_batch: torch.Tensor,
    xapps: Sequence[Any],
    weights: Sequence[float],
    scaling_term: float,
    device: torch.device | str,
) -> torch.Tensor:
    count = next_kpis_batch.shape[0]
    kpis_np = next_kpis_batch.cpu().detach().numpy()
    costs = np.zeros(count)
    for index in range(count):
        kpis = kpis_np[index]
        cost_vec = np.zeros(len(xapps))
        sat_vec = np.zeros(len(xapps))
        for xapp_index, xapp in enumerate(xapps):
            utility = xapp.compute_utility(kpis)
            distance, satisfied = weighted_distance(xapp, utility)
            cost_vec[xapp_index] = weights[xapp_index] * distance * scaling_term
            sat_vec[xapp_index] = satisfied
        costs[index] = cost_vec.sum() - (sat_vec.sum()) ** 2
    return torch.tensor(costs, dtype=torch.float32, device=device)
