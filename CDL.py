import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StatePredictor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, pred_hidden):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, hidden_dim, []) for _ in range(state_dim)]
        )
        self.action_feature_extractor = MLP(action_dim, hidden_dim, [])

        self.predictor = MLP(hidden_dim, 2, pred_hidden)

    def forward(self, s, a, mask=None):

        feats = []

        fa = self.action_feature_extractor(a)          
        feats.append(fa)

        for i in range(self.state_dim):
            fi = self.state_feature_extractors[i](s[:, i:i+1])  
            if mask is not None and mask[i] == 0:
                fi = torch.full_like(fi, -1e9)
            feats.append(fi)

        h = torch.stack(feats, dim=0).max(dim=0).values  

        out = self.predictor(h)
        mean = out[:, 0:1]
        std  = torch.clamp(torch.exp(out[:, 1:2]), 1e-3, 1e-2)
        return mean, std


class CDL:

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        pred_hidden=[64, 32],
        lr=3e-4,
        cmi_threshold=0.02,
        ema_decay=0.999,
        device=None,
    ):
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.cmi_threshold = cmi_threshold
        self.ema_decay     = ema_decay

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.models = nn.ModuleList([
            StatePredictor(state_dim, action_dim, hidden_dim, pred_hidden)
            for _ in range(state_dim)
        ]).to(self.device)
        self.opt = optim.Adam(self.models.parameters(), lr=lr)

        self.cmi_matrix = torch.zeros(state_dim, state_dim, device=self.device)


    def gaussian_nll(self, mean, std, target):
        """Gaussian negative log-likelihood (scalar per element)."""
        var = std ** 2
        return 0.5 * torch.log(2 * torch.pi * var) + (target - mean) ** 2 / (2 * var)

    def _parent_mask(self, j):
        """Binary mask that keeps only the inferred parents of s^j_{t+1}."""
        pa_mask = torch.zeros(self.state_dim, device=self.device)
        parents = (self.cmi_matrix[:, j] > self.cmi_threshold).nonzero(as_tuple=True)[0]
        pa_mask[parents] = 1
        return pa_mask


    def full_cdl_loss(self, s_batch, a_batch):

        s      = s_batch[:, 0]          # (B, D)
        s_next = s_batch[:, 1]          # (B, D)
        a      = a_batch                # (B, action_dim)

        loss = torch.tensor(0.0, device=self.device)

        for j in range(self.state_dim):
            target = s_next[:, j:j+1]   # (B, 1)

            # --- Term 1: full conditioning  p(s^j_{t+1} | a_t, s_t) ---
            mean, std = self.models[j](s, a, mask=None)
            loss = loss + self.gaussian_nll(mean, std, target).mean()

            # --- Term 2: mask out a random i != j ---
            # p(s^j_{t+1} | {a_t, s_t \ s^i_t})
            i = random.choice([idx for idx in range(self.state_dim) if idx != j])
            mask_i = torch.ones(self.state_dim, device=self.device)
            mask_i[i] = 0
            m_mean, m_std = self.models[j](s, a, mask=mask_i)
            loss = loss + self.gaussian_nll(m_mean, m_std, target).mean()

            # --- Term 3: parent-only conditioning  p(s^j_{t+1} | PA_{s^j}) ---
            pa_mask = self._parent_mask(j)
            p_mean, p_std = self.models[j](s, a, mask=pa_mask)
            loss = loss + self.gaussian_nll(p_mean, p_std, target).mean()

        return loss

    def train_step(self, s_batch, a_batch):
        """
        One gradient step.

        Args:
            s_batch : (B, 2, state_dim)
            a_batch : (B, action_dim)
        Returns:
            float loss value
        """
        self.opt.zero_grad()
        loss = self.full_cdl_loss(s_batch, a_batch)
        loss.backward()
        self.opt.step()
        return loss.item()

    # ------------------------------------------------------------------
    # CMI estimation  (Eq. 2 — validation data, no gradients)
    # ------------------------------------------------------------------

    def evaluate_cmi(self, s_val, a_val):
        """
        Estimates CMI_{ij} for all (i, j) pairs on validation data and
        updates self.cmi_matrix via EMA.

        CMI_{ij} = E[log p(s^j_{t+1}|a_t,s_t) - log p(s^j_{t+1}|{a_t,s_t\s^i_t})]
                 = E[NLL_masked - NLL_full]   (positive when i is causal for j)

        Args:
            s_val : (B, 2, state_dim)
            a_val : (B, action_dim)
        """
        s      = s_val[:, 0]
        s_next = s_val[:, 1]
        a      = a_val

        with torch.no_grad():
            for j in range(self.state_dim):
                target = s_next[:, j:j+1]

                # Full-input NLL (baseline)
                full_mean, full_std = self.models[j](s, a, mask=None)
                full_nll = self.gaussian_nll(full_mean, full_std, target).mean()

                for i in range(self.state_dim):
                    # Mask out s^i_t
                    mask_i = torch.ones(self.state_dim, device=self.device)
                    mask_i[i] = 0
                    m_mean, m_std = self.models[j](s, a, mask=mask_i)
                    masked_nll = self.gaussian_nll(m_mean, m_std, target).mean()

                    # CMI >= 0 when s^i_t is informative for s^j_{t+1}
                    # Clamp at 0 to avoid noise pushing CMI negative
                    cmi = torch.clamp(masked_nll - full_nll, min=0.0)

                    # EMA update
                    self.cmi_matrix[i, j] = (
                        self.ema_decay       * self.cmi_matrix[i, j]
                        + (1 - self.ema_decay) * cmi.item()
                    )

    # ------------------------------------------------------------------
    # Causal graph
    # ------------------------------------------------------------------

    def get_causal_graph(self):
        """
        Returns a (state_dim x state_dim) binary adjacency matrix where
        entry [i, j] = 1 iff the causal edge s^i_t -> s^j_{t+1} is inferred.
        """
        return (self.cmi_matrix > self.cmi_threshold).int()

    def to(self, device):
        """Move the entire CDL model to a new device."""
        self.device = torch.device(device)
        self.models = self.models.to(self.device)
        self.cmi_matrix = self.cmi_matrix.to(self.device)
        return self


# =============================================================================
# Usage example
# =============================================================================
#
#   cdl = CDL(state_dim=5, action_dim=2)
#   print(cdl.device)   # cuda:0 if GPU is available, else cpu
#
#   # Your data must be moved to the same device before passing in:
#   s_batch = s_batch.to(cdl.device)   # (B, 2, state_dim)
#   a_batch = a_batch.to(cdl.device)   # (B, action_dim)
#
#   loss = cdl.train_step(s_batch, a_batch)
#   cdl.evaluate_cmi(s_val.to(cdl.device), a_val.to(cdl.device))
#   print(cdl.get_causal_graph())
#
#   # To force a specific device:
#   cdl = CDL(state_dim=5, action_dim=2, device="cuda:0")
#   cdl = CDL(state_dim=5, action_dim=2, device="cpu")