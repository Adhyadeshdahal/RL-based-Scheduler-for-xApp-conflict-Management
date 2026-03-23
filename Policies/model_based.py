import numpy as np
import torch


class ModelBasedPolicy:

    def __init__(
        self,
        cdl,
        env,
        reward_fn,
        n_horizon   = 1,
        n_candidate = 500,
        n_top       = 50,
        n_iter      = 5,
    ):
        self.cdl         = cdl
        self.env         = env
        self.reward_fn   = reward_fn
        self.n_horizon   = n_horizon
        self.n_candidate = n_candidate
        self.n_top       = n_top
        self.n_iter      = n_iter
        self.device      = cdl.device
        self.num_bins    = env.num_bins
        self.num_params  = env.num_params
        self.kpi_start   = env.num_params
        self.state_dim   = cdl.state_dim
        self.action_dim  = cdl.action_dim

        self.action_space = env.action_space #

    def act(self, obs_tensor):

        s = obs_tensor.float().to(self.device)

        self.cdl.models.eval()
        with torch.no_grad():
            action = self._cem(s)
        self.cdl.models.train()

        return action


    def _cem(self, s0):
        # action_space is [num_params-1, num_bins-1, max_bin_length-1]
        action_high = torch.tensor(self.action_space, dtype=torch.float32, device=self.device)  # (action_dim,)
        
        mu  = action_high / 2.0                                          # (action_dim,)
        mu  = mu.unsqueeze(0).expand(self.n_horizon, -1).clone()         # (H, action_dim)
        std = mu.clone()                                                  # (H, action_dim)

        for _ in range(self.n_iter):
            noise   = torch.randn(self.n_candidate, self.n_horizon,
                                self.action_dim, device=self.device)
            actions = (mu + std * noise).round().long()                  # (n, H, action_dim)
            # clamp each dimension independently
            actions = torch.stack([
                actions[..., i].clamp(0, self.action_space[i])
                for i in range(self.action_dim)
            ], dim=-1)                                                    # (n, H, action_dim)

            pred_kpis = self._rollout(s0, actions)                       # (n, H, n_kpis)
            rewards   = self._score(pred_kpis, actions)                  # (n,)
            top_idx   = torch.argsort(rewards, descending=True)[:self.n_top]
            elite     = actions[top_idx].float()                         # (n_top, H, action_dim)
            mu        = elite.mean(dim=0)                                # (H, action_dim)
            std        = elite.std(dim=0).clamp(min=1e-3)               # (H, action_dim)

        best = mu[0].round().long()
        best = torch.stack([
            best[i].clamp(0, self.action_space[i])
            for i in range(self.action_dim)
        ])                                                               # (action_dim,)
        return best.cpu().numpy()                                        # shape (3,)


    def _rollout(self, s0, actions):
        n = self.n_candidate

        s = s0.unsqueeze(0).expand(n, -1).clone().float() 

        pred_kpis = []

        for h in range(self.n_horizon):
            a = actions[:, h, :].float()  
            print(a.shape)         #debug          

            kpi_dists = self.cdl.predictNextState(s, a)

            kpi_next = kpi_dists.sample()                                   # (n, n_kpis)

            pred_kpis.append(kpi_next)

            s = s.clone()
            s[:, self.kpi_start:] = kpi_next

        return torch.stack(pred_kpis, dim=1)                # (n, H, n_kpis)


    def _score(self, pred_kpis, actions):
        """
        pred_kpis: (n_candidate, n_horizon, n_kpis)
        actions:   (n_candidate, n_horizon, action_dim)
        Returns:   (n_candidate,)
        """
        rewards = self.reward_fn(pred_kpis, actions)        # (n, H)
        return rewards.sum(dim=-1)                          # (n,)