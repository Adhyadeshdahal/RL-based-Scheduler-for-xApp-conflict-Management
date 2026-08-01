from Algorithms.model_based_cem import ModelBasedCEM
from Algorithms.model_based_mcts import ModelBasedMCTS
from Algorithms.model_based_mppi import ModelBasedMPPI
from Algorithms.QACM import QACM
from config import ExperimentConfig


def get_algorithms(cfg: ExperimentConfig, model, env):
    qacm = QACM(model=model, env=env)
    cem = ModelBasedCEM(model=model, env=env, **vars(cfg.planner.cem))
    mppi = ModelBasedMPPI(model=model, env=env, **vars(cfg.planner.mppi))
    mcts = ModelBasedMCTS(model=model, env=env, **vars(cfg.planner.mcts))
    return [qacm, cem, mppi, mcts]
