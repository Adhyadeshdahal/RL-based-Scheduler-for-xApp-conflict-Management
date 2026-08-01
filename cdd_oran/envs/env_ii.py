from math import exp

from cdd_oran.config import DEFAULT_CONFIG
from cdd_oran.envs.base import BaseORANEnv
from cdd_oran.envs.statistics import get_env_ii_mean_std


def safe_exp(x):
    return x if abs(x) > 1e-1 else 1e-1


def update_kpi1(prev_params, prev_kpis):
    return 80 * exp(-((prev_params[0]) ** 2) / (2 * (safe_exp(prev_params[1]) ** 2)))


def update_kpi2(prev_params, prev_kpis):
    return 100 * exp(
        -((prev_params[0] + prev_params[2]) ** 2) / (2 * (safe_exp(prev_params[1]) ** 2))
    )


def update_kpi3(prev_params, prev_kpis):
    return 120 * exp(-((prev_params[0] + 45) ** 2) / (2 * (safe_exp(prev_params[3]) ** 2)))


def update_kpi41(prev_params, prev_kpis):
    return 120 * exp(
        -((prev_params[5] + prev_params[1] - 30) ** 2) / (2 * (safe_exp(prev_params[4]) ** 2))
    )


def update_kpi42(prev_params, prev_kpis):
    return 150 * exp(
        -((prev_params[5] + prev_params[1] - 50) ** 2) / (2 * (safe_exp(prev_params[4]) ** 2))
    )


def update_kpi5(prev_params, prev_kpis):
    return -35 * exp(
        -((prev_params[7] + prev_params[0] - 25) ** 2) / (2 * (safe_exp(prev_params[6]) ** 2))
    )


def _param_ranges(cfg):
    if cfg.param_ranges == "ood":
        return [
            (100, 150),
            (50, 100),
            (-30, 30),
            (-90, 90),
            (-30, -19),
            (-50, 150),
            (66, 87),
            (-200, 150),
        ]
    return [
        (-100, 100),
        (-10, 50),
        (-20, 20),
        (-60, 60),
        (-20, 20),
        (-50, 150),
        (-60, 65),
        (-100, 150),
    ]


class ORANEnvironment2(BaseORANEnv):
    def __init__(self, num_bins=10, max_steps=50, cfg=DEFAULT_CONFIG):
        param_ranges = _param_ranges(cfg)
        super().__init__(
            param_ranges=param_ranges,
            mean_std_kpis=get_env_ii_mean_std(param_ranges, seed=cfg.seed),
            kpi_thresholds=[55, 95, 85, 75, 80, -25],
            kpi_names=["kpi1", "kpi2", "kpi3", "kpi41", "kpi42", "kpi5"],
            directions=[0, 0, 0, 0, 0, 1],
            update_fns=[
                update_kpi1,
                update_kpi2,
                update_kpi3,
                update_kpi41,
                update_kpi42,
                update_kpi5,
            ],
            xapp_param_indices=[(0, 1), (0, 1, 2), (0, 3), (4, 5, 1), (0, 6, 7)],
            xapp_kpi_indices=[(0,), (1,), (2,), (3, 4), (5,)],
            kpi_to_xapp={0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4},
            adjacency_edges=[
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 3),
                (2, 0),
                (3, 1),
                (3, 4),
                (3, 5),
                (4, 1),
                (4, 4),
                (4, 5),
                (5, 0),
                (5, 6),
                (5, 7),
            ],
            num_bins=num_bins,
            max_steps=max_steps,
        )
