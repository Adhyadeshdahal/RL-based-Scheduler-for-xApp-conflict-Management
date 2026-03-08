import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


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

    def __init__(self, state_dim, action_dim, feature_dim, pred_hidden):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # state feature extractors — one per variable
        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, feature_dim, []) for _ in range(state_dim)]
        )

        # action feature extractor
        self.action_feature_extractor = MLP(action_dim, feature_dim, [])

        # predictor
        self.predictor = MLP(feature_dim, 2, pred_hidden)

    def forward(self, s, a, mask=None):
        """
        s:    (bs, state_dim)
        a:    (bs, action_dim)
        mask: (bs, state_dim+1) bool — True = mask out this parent
        returns: Normal distribution of shape (bs, state_dim)
        """
        bs = s.shape[0]
        fd = self.state_dim

        # extract state features
        feats = []
        for i in range(fd):
            f_i = self.state_feature_extractors[i](s[:, i:i+1])  # (bs, feature_dim)
            feats.append(f_i)

        # extract action feature
        a_feat = self.action_feature_extractor(a)                 # (bs, feature_dim)
        feats.append(a_feat)

        feats = torch.stack(feats, dim=1)                         # (bs, fd+1, feature_dim)

        # mask before max-pool
        if mask is not None:
            feats = feats.masked_fill(mask.unsqueeze(-1), float('-inf'))

        # max-pool over parents
        h, _ = feats.max(dim=1)                                   # (bs, feature_dim)

        # predict mu, std
        out     = self.predictor(h)                               # (bs, 2)
        mu      = out[:, 0:1]                                     # (bs, 1)
        log_std = out[:, 1:2]
        std     = torch.exp(torch.clamp(log_std, -5, 2)) + 1e-4

        return mu, std


class CDL:

    def __init__(
        self,
        state_dim,
        action_dim,
        feature_fc_dims=(64,),
        generative_fc_dims=(64, 32),
        lr=3e-4,
        cmi_threshold=0.2,
        eval_tau=0.99,
        eval_steps=10,
        grad_clip=10.0,
        device=None,
    ):
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.cmi_threshold = cmi_threshold
        self.eval_tau      = eval_tau
        self.eval_steps    = eval_steps
        self.grad_clip     = grad_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        feature_dim = feature_fc_dims[-1]

        # one predictor per child variable
        self.models = nn.ModuleList([
            StatePredictor(state_dim, action_dim, feature_dim, list(generative_fc_dims))
            for _ in range(state_dim)
        ]).to(self.device)

        self.opt = optim.Adam(self.models.parameters(), lr=lr)

        # CMI matrix (state_dim, state_dim+1) — last col is action
        fd = state_dim
        self.mask_CMI = torch.zeros(fd, fd + 1, device=self.device)

        # accumulators for periodic EMA update
        self._eval_cmi_acc    = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_step_count = 0

    def _nll(self, mu, std, target):
        return -Normal(mu, std).log_prob(target)                  # (bs, 1)

    def train_step(self, s_batch, a_batch):
        """
        s_batch: (bs, 2, state_dim)
        a_batch: (bs, action_dim)
        """
        s_t   = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs    = s_t.shape[0]
        fd    = self.state_dim

        self.opt.zero_grad()
        total_loss = 0.0

        # random parent to drop — one per sample
        drop_idx = torch.randint(fd + 1, (bs,), device=self.device)
        mask     = F.one_hot(drop_idx, fd + 1).bool()             # (bs, fd+1)

        for j in range(fd):
            target = s_tp1[:, j:j+1]                              # (bs, 1)

            # full prediction
            mu, std = self.models[j](s_t, a_batch)
            full_loss = self._nll(mu, std, target).mean()

            # masked prediction
            mu_m, std_m = self.models[j](s_t, a_batch, mask=mask)
            masked_loss = self._nll(mu_m, std_m, target).mean()

            total_loss += full_loss + masked_loss

        loss = total_loss / fd
        loss.backward()
        nn.utils.clip_grad_norm_(self.models.parameters(), self.grad_clip)
        self.opt.step()

        return loss.item()

    def update_mask(self, s_batch, a_batch):
        """
        s_batch: (bs, 2, state_dim)
        a_batch: (bs, action_dim)
        Accumulates CMI over eval_steps calls then does one EMA update.
        """
        s_t   = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs    = s_t.shape[0]
        fd    = self.state_dim

        step_cmi = torch.zeros(fd, fd + 1, device=self.device)

        with torch.no_grad():
            for j in range(fd):
                target = s_tp1[:, j:j+1]                          # (bs, 1)

                # full prediction for child j
                mu, std   = self.models[j](s_t, a_batch)
                full_nll  = self._nll(mu, std, target)             # (bs, 1)

                # drop each parent one at a time
                for i in range(fd + 1):
                    mask = torch.zeros(bs, fd + 1, dtype=torch.bool, device=self.device)
                    mask[:, i] = True

                    mu_m, std_m  = self.models[j](s_t, a_batch, mask=mask)
                    masked_nll   = self._nll(mu_m, std_m, target)  # (bs, 1)

                    # CMI[j, i] = how much does dropping parent i hurt child j
                    step_cmi[j, i] = (masked_nll - full_nll).mean()

        self._eval_cmi_acc    += step_cmi
        self._eval_step_count += 1

        if self._eval_step_count >= self.eval_steps:
            avg_cmi       = self._eval_cmi_acc / self.eval_steps
            self.mask_CMI = self.eval_tau * self.mask_CMI + (1 - self.eval_tau) * avg_cmi

            self._eval_cmi_acc    = torch.zeros(fd, fd + 1, device=self.device)
            self._eval_step_count = 0

    def get_causal_graph(self):
        """Raw CMI values (fd, fd+1) — last col is action."""
        return self.mask_CMI

    def get_binary_graph(self):
        """Thresholded binary graph (fd, fd+1)."""
        graph = (self.mask_CMI >= self.cmi_threshold)
        fd = graph.shape[0]
        if graph.shape[1] > fd:
            graph[:fd, :fd].fill_diagonal_(0)
        return graph