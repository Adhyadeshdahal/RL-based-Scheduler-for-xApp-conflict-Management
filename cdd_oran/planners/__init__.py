from cdd_oran.config import ExperimentConfig
from cdd_oran.planners.cem import ModelBasedCEM
from cdd_oran.planners.mcts import ModelBasedMCTS
from cdd_oran.planners.mppi import ModelBasedMPPI
from cdd_oran.planners.qacm import QACM


def get_planners(cfg: ExperimentConfig, model, env):
    qacm = QACM(model=model, env=env)
    cem = ModelBasedCEM(model=model, env=env, **vars(cfg.planner.cem))
    mppi = ModelBasedMPPI(model=model, env=env, **vars(cfg.planner.mppi))
    mcts = ModelBasedMCTS(model=model, env=env, **vars(cfg.planner.mcts))
    return [qacm, cem, mppi, mcts]


__all__ = ["get_planners", "ModelBasedCEM", "ModelBasedMCTS", "ModelBasedMPPI", "QACM"]
