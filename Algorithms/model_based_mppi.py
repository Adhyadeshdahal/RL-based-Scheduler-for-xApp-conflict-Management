import numpy as np
import torch
from Parameters import *


class ModelBasedMPPI:
    """
    Model Predictive Path Integral planning.
    Same act() interface as QACM and ModelBasedCEM.

    Key difference from CEM: all N samples contribute to the update
    via softmax weights — no hard elite selection, no iteration loop.
    One forward pass through the model is enough.
    """

    def __init__(
        self,
        model,
        env,
        n_samples = N_SAMPLES,
        temperature = TEMPERATURE,   
        noise_sigma = NOISE_SIGMA,   
    ):
        self.model       = model
        self.env         = env
        self.xapps       = env.xapps
        self.num_bins    = env.num_bins
        self.num_params  = env.num_params
        self.action_space = env.action_space
        self.name = "ModelBasedMPPI"

        self.n_samples   = n_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.device      = model.device


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

        hi = torch.tensor(
            [self.action_space[1], self.action_space[2]],
            dtype=torch.float32, device=self.device
        )
        mu = hi / 2.0   # (2,)

        # ── 1. Sample perturbations around nominal ─────────────────────
        noise   = torch.randn(self.n_samples, 2, device=self.device) * self.noise_sigma
        samples = (mu + noise).round().long()          # (N, 2)
        samples = torch.stack([
            samples[:, 0].clamp(0, self.action_space[1]),
            samples[:, 1].clamp(0, self.action_space[2]),
        ], dim=1)                                       # (N, 2)

        # ── 2. Batch predict next KPIs ─────────────────────────────────
        pi_col       = torch.full((self.n_samples, 1), pi,
                                   dtype=torch.long, device=self.device)
        action_batch = torch.cat([pi_col, samples], dim=1).float()   # (N, 3)

        s_batch          = s0.unsqueeze(0).expand(self.n_samples, -1).float()
        next_state_dist  = self.model.predictNextState(s_batch, action_batch)
        next_kpis_batch  = next_state_dist.mean                       # (N, n_kpis)

        # ── 3. Score every sample ──────────────────────────────────────
        costs = self._score_batch(next_kpis_batch, xapps, w, tau)    # (N,)

        # ── 4. MPPI weight: softmax over negative cost / temperature ───
        #    exp(-cost/λ) normalised — lower cost → higher weight
        log_weights = -costs / self.temperature
        log_weights -= log_weights.max()               # numerical stability
        weights     = torch.exp(log_weights)
        weights     = weights / weights.sum()          # (N,)  sums to 1

        # ── 5. Weighted average → best action ─────────────────────────
        best_2d = (weights.unsqueeze(1) * samples.float()).sum(dim=0)  # (2,)
        best_bin = best_2d[0].round().long().clamp(0, self.action_space[1]).item()
        best_idx = best_2d[1].round().long().clamp(0, self.action_space[2]).item()
        return [pi, best_bin, best_idx]


    def _score_batch(self, next_kpis_batch, xapps, w, tau):
        N       = next_kpis_batch.shape[0]
        kpis_np = next_kpis_batch.cpu().detach().numpy()
        costs   = np.zeros(N)
        for i in range(N):
            kpis     = kpis_np[i]
            cost_vec = np.zeros(len(xapps))
            sat_vec  = np.zeros(len(xapps))
            for j, xapp in enumerate(xapps):
                u           = xapp.compute_utility(kpis)
                d, s        = self._weighted_distance(xapp, u)
                cost_vec[j] = w[j] * d * tau
                sat_vec[j]  = s
            costs[i] = cost_vec.sum() - (sat_vec.sum()) ** 2
        return torch.tensor(costs, dtype=torch.float32, device=self.device)

    @staticmethod
    def _weighted_distance(xapp, utility):
        if xapp.direction == 0:
            if utility < xapp.threshold:
                return xapp.threshold - utility, 0
            return 0.0, 1
        else:
            if utility > xapp.threshold:
                return utility - xapp.threshold, 0
            return 0.0, 1