import numpy as np
import torch


class QACM:
    def __init__(self, model, env):
        self.model        = model
        self.env          = env
        self.xapps        = env.xapps
        self.num_bins     = env.num_bins
        self.num_params   = env.num_params
        self.action_space = env.action_space  # [num_params-1, num_bins-1, *max_bin_lengths]
        self.name         = "QACM"

    def compute_utility(self, xapp, kpis):
        return xapp.compute_utility(kpis)

    def obtain_weighted_distance(self, xapp, utility):
        # FIX 2: threshold must be in the same normalised z-score space as utility.
        # xapp.threshold is a raw KPI value; normalise it before comparing.
        # Compare utility (z-score) against the normalised threshold.
        # Since env sets xapp.threshold = KPI_THRESHOLDS[i] (raw), we normalise here.
        kpi_idx = self.env.xapps.index(xapp)
        mean, std = self.env.kpis[kpi_idx].mean, self.env.kpis[kpi_idx].std
        norm_threshold = (xapp.threshold - mean) / std

        d, s = 0, 0
        if xapp.direction == 0:          # maximise
            if utility < norm_threshold:
                d = norm_threshold - utility
                s = 0
            else:
                d = 0.0
                s = 1
        else:                            # minimise
            if utility > norm_threshold:
                d = utility - norm_threshold
                s = 0
            else:
                d = 0.0
                s = 1
        return d, s

    def act(self, current_state, conflict_param_index,
            xapps_under_conflict, weights_per_xapps, scaling_term):

        s0    = current_state
        pi    = conflict_param_index
        xapps = xapps_under_conflict
        w     = weights_per_xapps
        tau   = scaling_term

        # FIX 3: use per-param bin-length as upper bound for index, not action_space[2]
        # action_space = [num_params-1, num_bins-1] + max_bin_length (per-param list)
        max_index = self.env.action_space[pi + 2]

        pl_opt   = None
        min_cost = float('inf')

        for bin_id in range(self.action_space[1] + 1):
            for index in range(max_index + 1):          # FIX 3
                action = [pi, bin_id, index]

                action_tensor = torch.tensor(
                    action, dtype=torch.float32
                ).unsqueeze(0).to(self.model.device)

                next_state_dist = self.model.predictNextState(
                    s0.unsqueeze(0), action_tensor
                )
                # Model returns KPI portion only
                next_state = next_state_dist.sample().squeeze(0)
                next_kpis  = next_state.cpu().numpy()

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