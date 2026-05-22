import numpy as np
import torch
from Parameters import *


class ModelBasedCEM:
    """
    CEM-based conflict resolution — same act() interface as QACM,
    but replaces the brute-force grid search with iterative
    cross-entropy method planning over (bin_id, index) pairs.
    """

    def __init__(
        self,
        model,
        env,
        n_candidate: int = N_CANDIDATE,
        n_top:       int = N_TOP,
        n_iter:      int = N_ITER,
    ):
        self.model        = model
        self.env          = env
        self.xapps        = env.xapps
        self.num_bins     = env.num_bins
        self.num_params   = env.num_params
        self.action_space = env.action_space
        self.name         = 'ModelBasedCEM'

        self.n_candidate  = n_candidate
        self.n_top        = n_top
        self.n_iter       = n_iter
        self.device       = model.device

    def act(
        self,
        current_state,
        conflict_param_index,
        xapps_under_conflict,
        weights_per_xapps,
        scaling_term,
    ):
        s0    = current_state
        pi    = conflict_param_index
        xapps = xapps_under_conflict
        w     = weights_per_xapps
        tau   = scaling_term

        # FIX 3: use per-param bin-length as upper bound for index dimension
        max_index = self.env.action_space[pi + 2]

        lo = torch.zeros(2, device=self.device)
        hi = torch.tensor(
            [self.action_space[1], max_index],   # FIX 3
            dtype=torch.float32, device=self.device
        )

        mu  = (lo + hi) / 2.0
        std = (hi - lo) / 6.0

        for _ in range(self.n_iter):
            noise   = torch.randn(self.n_candidate, 2, device=self.device)
            samples = (mu + std * noise).round().long()
            samples = torch.stack([
                samples[:, 0].clamp(0, self.action_space[1]),
                samples[:, 1].clamp(0, max_index),            # FIX 3
            ], dim=1)

            pi_col       = torch.full((self.n_candidate, 1), pi,
                                      dtype=torch.long, device=self.device)
            action_batch = torch.cat([pi_col, samples], dim=1).float()

            s_batch         = s0.unsqueeze(0).expand(self.n_candidate, -1).float()
            next_state_dist = self.model.predictNextState(s_batch, action_batch)
            next_state_batch = next_state_dist.sample()

            # Model returns KPI portion only
            next_kpis_batch = next_state_batch

            scores = self._score_batch(next_kpis_batch, action_batch, xapps, w, tau, max_index)

            top_idx = torch.argsort(scores)[:self.n_top]
            elites  = samples[top_idx].float()
            mu      = elites.mean(dim=0)
            std     = elites.std(dim=0).clamp(min=1.0)   # FIX: min=1.0 for discrete space

        best_bin = mu[0].round().long().clamp(0, self.action_space[1]).item()
        best_idx = mu[1].round().long().clamp(0, max_index).item()         # FIX 3
        return [pi, best_bin, best_idx]

    def _score_batch(self, next_kpis_batch, action_batch, xapps, w, tau, max_index):
        """
        next_kpis_batch : (N, n_kpis) — already sliced to KPI portion only
        Returns         : (N,) float tensor  (lower = better)
        """
        N       = next_kpis_batch.shape[0]
        kpis_np = next_kpis_batch.cpu().detach().numpy()

        costs = np.zeros(N)
        for i in range(N):
            kpis     = kpis_np[i]
            cost_vec = np.zeros(len(xapps))
            sat_vec  = np.zeros(len(xapps))

            for j, xapp in enumerate(xapps):
                u           = xapp.compute_utility(kpis)
                d, s        = self._weighted_distance(xapp, u, j)
                cost_vec[j] = w[j] * d * tau
                sat_vec[j]  = s

            f_cost = cost_vec.sum() - (sat_vec.sum()) ** 2
            costs[i] = f_cost

        return torch.tensor(costs, dtype=torch.float32, device=self.device)

    def _weighted_distance(self, xapp, utility, xapp_idx):
        # FIX 2: normalise raw KPI threshold into z-score space before comparing
        mean, std  = xapp.mean, xapp.std
        norm_threshold = (xapp.threshold - mean) / std

        if xapp.direction == 0:           # maximise
            if utility < norm_threshold:
                return norm_threshold - utility, 0
            return 0.0, 1
        else:                             # minimise
            if utility > norm_threshold:
                return utility - norm_threshold, 0
            return 0.0, 1