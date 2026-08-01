from typing import Any

from config import ExperimentConfig
from Environment.Environment_I import ORANEnvironment
from Environment.Environment_II import ORANEnvironment2


def get_env(cfg: ExperimentConfig):
    env: Any = None
    if cfg.environment == "EnvironmentII":
        env = ORANEnvironment2(cfg=cfg)
    elif cfg.environment == "EnvironmentI":
        env = ORANEnvironment(cfg=cfg)
    else:
        raise NameError(f"env{env} is not valid")
    return env
