import numpy as np
import torch

class QACM:
    def __init__(self, model, env):
        self.model      = model
        self.env        = env
        self.xapps      = env.xapps
        self.num_bins   = env.num_bins
        self.num_params = env.num_params
        self.action_space = env.action_space  # [num_params-1, num_bins-1, num_bins-1]
        self.name = "QACM"

    def compute_utility(self, xapp, kpis):
        return xapp.compute_utility(kpis)

    def obtain_weighted_distance(self, xapp, utility):
        d, s = 0, 0
        if xapp.direction == 0:
            if utility < xapp.threshold:
                d = xapp.threshold - utility
                s = 0
            else:
                d = 0
                s = 1
        else:
            if utility > xapp.threshold:
                d = utility - xapp.threshold
                s = 0
            else:
                d = 0
                s = 1
        return d, s

    def act(self, current_state, conflict_param_index,
            xapps_under_conflict, weights_per_xapps, scaling_term):

        s0      = current_state
        pi      = conflict_param_index
        xapps   = xapps_under_conflict
        w       = weights_per_xapps
        tau     = scaling_term

        pl_opt   = None
        min_cost = float('inf')

        for bin_id in range(self.action_space[1] + 1):
            for index in range(self.action_space[2] + 1):
                action = [pi, bin_id, index]

                action_tensor = torch.tensor(
                    action, dtype=torch.float32
                ).unsqueeze(0).to(self.model.device)          # (1, 3)

                next_state_dist = self.model.predictNextState(
                    s0.unsqueeze(0), action_tensor
                )
                next_kpis = next_state_dist.mean.squeeze(0).cpu().numpy()  # (n_kpis,)

                cost = np.zeros(len(xapps))
                s    = np.zeros(len(xapps))

                for i, xapp in enumerate(xapps):
                    u_i      = self.compute_utility(xapp, next_kpis)
                    d_i, s_i = self.obtain_weighted_distance(xapp, u_i)
                    cost[i]  = w[i] * d_i * tau
                    s[i]     = s_i

                f_cost = cost.sum() - (s.sum()) ** 2
                if f_cost < min_cost:
                    min_cost = f_cost
                    pl_opt   = action 

        return pl_opt  