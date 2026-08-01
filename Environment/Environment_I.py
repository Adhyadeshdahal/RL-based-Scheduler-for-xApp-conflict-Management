from math import exp

from config import DEFAULT_CONFIG
from env_statistics import get_envI_mean_std
from Environment.base import BaseORANEnv


def update_Kpi1(prev_params, prev_kpis):
    P1 = prev_params[0]
    P2 = max(prev_params[1], 1e-3)
    return 0.5 * exp(-((P1 + 50) ** 2) / (2 * P2) ** 2)


def update_Kpi2(prev_params, prev_kpis):
    P1 = prev_params[0]
    P3 = max(prev_params[2], 1e-3)
    return exp(-((P1 - 50) ** 2) / (2 * P3) ** 2)


def update_Kpi3(prev_params, prev_kpis):
    P4 = prev_params[3]
    P5 = max(prev_params[4], 1e-3)
    return exp(-((P4 + prev_kpis[0]) ** 2) / (2 * P5) ** 2)


def update_Kpi4(prev_params, prev_kpis):
    P7 = prev_params[6]
    P6 = max(prev_params[5], 1e-3)
    return exp(-((P7 + prev_kpis[1]) ** 2) / (2 * P6) ** 2)


def _param_ranges(cfg):
    if cfg.param_ranges == "ood":
        return [(0, 300), (0, 300), (0, 3), (0, 3), (0, 3), (0, 3), (0, 3)]
    return [(0, 300), (0, 300), (0, 3), (0, 3), (0, 3), (0, 3), (0, 3)]


class ORANEnvironment(BaseORANEnv):
    def __init__(self, num_bins=10, max_steps=50, cfg=DEFAULT_CONFIG):
        param_ranges = _param_ranges(cfg)
        super().__init__(
            param_ranges=param_ranges,
            mean_std_kpis=get_envI_mean_std(param_ranges, seed=cfg.seed),
            kpi_thresholds=[0.2, 0.6, 0.5, 0.5],
            kpi_names=["kpi1", "kpi2", "kpi3", "kpi4"],
            directions=[0, 0, 0, 0],
            update_fns=[update_Kpi1, update_Kpi2, update_Kpi3, update_Kpi4],
            xapp_param_indices=[(0, 1), (0, 2), (3, 4), (5, 6)],
            xapp_kpi_indices=[(0,), (1,), (2,), (3,)],
            kpi_to_xapp={0: 0, 1: 1, 2: 2, 3: 3},
            adjacency_edges=[
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 3),
                (2, 4),
                (2, 7),
                (3, 5),
                (3, 6),
                (3, 8),
            ],
            num_bins=num_bins,
            max_steps=max_steps,
        )
