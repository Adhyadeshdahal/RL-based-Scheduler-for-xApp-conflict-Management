import numpy as np

class RandomPolicy:
    def __init__(self, action_dim, action_space):
        self.action_space = action_space
        self.action_dim   = action_dim

    def act(self):
        param_id = np.random.randint(0, self.action_space[0] + 1)
        bin_id   = np.random.randint(0, self.action_space[1] + 1)
        index    = np.random.randint(0, self.action_space[2] + 1)
        return [param_id, bin_id, index]