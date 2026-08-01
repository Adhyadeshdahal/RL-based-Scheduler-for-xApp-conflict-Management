from Environment.Environment_I import ORANEnvironment
from Environment.Environment_II import ORANEnvironment2
from config import DEFAULT_CONFIG, ExperimentConfig


def _check_env_globals(cfg: ExperimentConfig):
    """Environments still read ranges and stats from Parameters/DEFAULT_CONFIG rather
    than from cfg, so these fields would be silently ignored. Removed in Phase 2."""
    for field in ("param_ranges", "seed"):
        got, supported = getattr(cfg, field), getattr(DEFAULT_CONFIG, field)
        if got != supported:
            raise NotImplementedError(
                f"cfg.{field}={got!r} would be ignored by the environment "
                f"(it uses {supported!r} from DEFAULT_CONFIG)."
            )


def get_env(cfg: ExperimentConfig):
    _check_env_globals(cfg)
    env = None
    if cfg.environment == "EnvironmentII":
        env = ORANEnvironment2()
    elif cfg.environment == "EnvironmentI":
        env = ORANEnvironment()
    else:
        raise NameError(f"env{env} is not valid")
    return env
