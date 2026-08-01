from Algorithms.QACM import QACM
from Algorithms.model_based_cem import ModelBasedCEM
from Algorithms.model_based_mppi import ModelBasedMPPI
from Algorithms.model_based_mcts import ModelBasedMCTS


def get_algorithms(model, env):
    qacm = QACM(model=model, env=env)
    cem = ModelBasedCEM(model=model, env=env)
    mppi = ModelBasedMPPI(model=model, env=env)
    mcts = ModelBasedMCTS(model=model, env=env)
    return [qacm, cem, mppi, mcts]
