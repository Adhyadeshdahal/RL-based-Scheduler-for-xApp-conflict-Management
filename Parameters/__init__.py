from config import DEFAULT_CONFIG
from env_statistics import get_envI_mean_std, get_envII_mean_std


SEED = DEFAULT_CONFIG.seed
ENVIRONMENT = DEFAULT_CONFIG.environment

if DEFAULT_CONFIG.param_ranges == "ood":
    ENVIRONMENT_II_PARAM_RANGES = [
        (100, 150),
        (50, 100),
        (-30, 30),
        (-90, 90),
        (-30, -19),
        (-50, 150),
        (66, 87),
        (-200, 150),
    ]
else:
    ENVIRONMENT_II_PARAM_RANGES = [
        (-100, 100),
        (-10, 50),
        (-20, 20),
        (-60, 60),
        (-20, 20),
        (-50, 150),
        (-60, 65),
        (-100, 150),
    ]

if DEFAULT_CONFIG.param_ranges == "ood":
    ENVIRONMENT_I_PARAM_RANGES = [
        (0, 300),
        (0, 300),
        (0, 3),
        (0, 3),
        (0, 3),
        (0, 3),
        (0, 3),
    ]
else:
    ENVIRONMENT_I_PARAM_RANGES = [
        (0, 300),
        (0, 300),
        (0, 3),
        (0, 3),
        (0, 3),
        (0, 3),
        (0, 3),
    ]


def __getattr__(name):
    if name == "ENVIRONMENT_I_MEAN_STDS":
        value = get_envI_mean_std(ENVIRONMENT_I_PARAM_RANGES, seed=SEED)
    elif name == "ENVIRONMENT_II_MEAN_STDS":
        value = get_envII_mean_std(ENVIRONMENT_II_PARAM_RANGES, seed=SEED)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
