from torch import Tensor
from random import random
import numpy as np
class RandomPolicy:
    def __init__(
        self,action_dim,num_bins,num_params,num_kpis
    ):
        self.n_params      = num_params
        self.n_kpis        = num_kpis
        self.max_action =    (num_params*num_bins) - 1

    def act(self):
        return np.random.randint(0, self.max_action)