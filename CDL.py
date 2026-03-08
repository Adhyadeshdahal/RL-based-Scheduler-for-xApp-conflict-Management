import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal


def reset_layer(weight, bias):
    nn.init.xavier_uniform_(weight)
    nn.init.zeros_(bias)


def forward_network(x, weights, biases, activation=F.relu):
    """
    x: (batch_leading_dims..., in_dim)
    weights: list of (batch_leading_dims..., in_dim, out_dim)
    biases:  list of (batch_leading_dims..., 1, out_dim)
    """
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = torch.bmm(x.unsqueeze(-2), w).squeeze(-2) + b.squeeze(-2)
        if activation is not None and i < len(weights) - 1:
            x = activation(x)
    return x


class CDL:
    """
    Causal Discovery via Likelihood (CDL) — mirrors InferenceCMI architecture.

    Key architectural points matching the original:
    - Shared action feature extractor weights across all child variables (feature_dim, in, out)
    - Shared state feature extractor weights across all (child, parent) pairs (feature_dim*feature_dim, in, out)
    - Max-pooling over parent features to get global feature
    - Masking with -inf before max-pool (not zeros)
    - CMI = masked_loss - full_loss, accumulated then EMA-updated periodically
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        feature_fc_dims=(64,),
        generative_fc_dims=(64, 32),
        lr=1e-4,
        cmi_threshold=0.02,
        eval_tau=0.99,
        eval_steps=10,
        grad_clip=10.0,
        device=None,
    ):
        self.state_dim = state_dim          # = feature_dim in original
        self.action_dim = action_dim
        self.feature_fc_dims = feature_fc_dims
        self.generative_fc_dims = generative_fc_dims
        self.cmi_threshold = cmi_threshold
        self.eval_tau = eval_tau
        self.eval_steps = eval_steps
        self.grad_clip = grad_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        fd = state_dim  # feature_dim alias

        # ---- action feature extractor ----
        # shape: (fd, in_dim, out_dim) per layer
        self.action_feature_weights = nn.ParameterList()
        self.action_feature_biases  = nn.ParameterList()
        in_dim = action_dim
        for out_dim in feature_fc_dims:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(fd, in_dim, out_dim)))
            self.action_feature_biases.append( nn.Parameter(torch.zeros(fd, 1,      out_dim)))
            in_dim = out_dim

        # ---- state feature extractor ----
        # shape: (fd*fd, in_dim, out_dim) per layer — one set of weights per (child, parent) pair
        self.state_feature_weights = nn.ParameterList()
        self.state_feature_biases  = nn.ParameterList()
        in_dim = 1  # each state variable is scalar
        for out_dim in feature_fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(fd * fd, in_dim, out_dim)))
            self.state_feature_biases.append( nn.Parameter(torch.zeros(fd * fd, 1,      out_dim)))
            in_dim = out_dim

        # ---- generative / predictor ----
        # shape: (fd, in_dim, out_dim) per layer, final layer outputs (mu, log_std) → 2
        self.generative_weights = nn.ParameterList()
        self.generative_biases  = nn.ParameterList()
        in_dim = feature_fc_dims[-1]
        for out_dim in generative_fc_dims:
            self.generative_weights.append(nn.Parameter(torch.zeros(fd, in_dim, out_dim)))
            self.generative_biases.append( nn.Parameter(torch.zeros(fd, 1,      out_dim)))
            in_dim = out_dim
        self.generative_weights.append(nn.Parameter(torch.zeros(fd, in_dim, 2)))
        self.generative_biases.append( nn.Parameter(torch.zeros(fd, 1,      2)))

        # move all parameters to device & initialise
        all_param_lists = [
            self.action_feature_weights, self.action_feature_biases,
            self.state_feature_weights,  self.state_feature_biases,
            self.generative_weights,     self.generative_biases,
        ]
        for pl in all_param_lists:
            for p in pl:
                p.data = p.data.to(self.device)

        self._reset_params()

        params = (
            list(self.action_feature_weights) + list(self.action_feature_biases) +
            list(self.state_feature_weights)  + list(self.state_feature_biases)  +
            list(self.generative_weights)     + list(self.generative_biases)
        )
        self.opt = optim.Adam(params, lr=lr)

        # ---- causal graph state ----
        fd1 = state_dim + 1
        self.mask_CMI = torch.zeros(fd, fd1, device=self.device)
        self.mask     = torch.zeros(fd, fd1, dtype=torch.bool, device=self.device)

        # diagonal: each variable always depends on itself
        diag = torch.eye(fd, fd1, dtype=torch.bool, device=self.device)
        self.mask[diag] = True

        # accumulators for periodic EMA update
        self._eval_cmi_acc   = torch.zeros(fd, fd1, device=self.device)
        self._eval_step_count = 0

    # ------------------------------------------------------------------
    # parameter initialisation
    # ------------------------------------------------------------------

    def _reset_params(self):
        fd = self.state_dim
        for w, b in zip(self.action_feature_weights, self.action_feature_biases):
            for i in range(fd):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_weights, self.state_feature_biases):
            for i in range(fd * fd):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_weights, self.generative_biases):
            for i in range(fd):
                reset_layer(w[i], b[i])

    # ------------------------------------------------------------------
    # feature extraction — mirrors original exactly
    # ------------------------------------------------------------------

    def _extract_action_feature(self, action):
        """
        action: (bs, action_dim)
        returns: (fd, 1, bs, out_dim)
        """
        fd = self.state_dim
        x = action.unsqueeze(0).expand(fd, -1, -1)          # (fd, bs, action_dim)
        # batched linear: for each of the fd networks
        out = self._forward_net_3d(x, self.action_feature_weights, self.action_feature_biases)
        return out.unsqueeze(1)                               # (fd, 1, bs, out_dim)

    def _extract_state_feature(self, state):
        """
        state: (bs, fd)
        returns: (fd, fd, bs, out_dim)
            first fd = child variable being predicted
            second fd = parent variable providing input
        """
        fd = self.state_dim
        bs = state.shape[0]

        x = state.t()                                         # (fd, bs)
        x = x.repeat(fd, 1, 1)                               # (fd, fd, bs)
        x = x.view(fd * fd, bs, 1)                           # (fd*fd, bs, 1)

        out = self._forward_net_3d(x, self.state_feature_weights, self.state_feature_biases)
        return out.view(fd, fd, bs, -1)                       # (fd, fd, bs, out_dim)

    def _predict_from_sa_feature(self, sa_feature, state):
        """
        sa_feature: (fd, bs, out_dim)
        state:      (bs, fd)   — used as residual base (not used here, kept for API parity)
        returns:    Normal dist of shape (bs, fd)
        """
        x = self._forward_net_3d(sa_feature, self.generative_weights, self.generative_biases,
                                  final_activation=False)     # (fd, bs, 2)
        x = x.permute(1, 0, 2)                               # (bs, fd, 2)
        mu, log_std = x.unbind(dim=-1)                       # (bs, fd) each
        std = torch.exp(torch.clamp(log_std, -5, 2)) + 1e-4
        return Normal(mu, std)

    # ------------------------------------------------------------------
    # forward pass  (mirrors forward_step in original)
    # ------------------------------------------------------------------

    def _forward(self, state, action, mask=None, action_feature=None, state_feature=None):
        """
        state:          (bs, fd)
        action:         (bs, action_dim)
        mask:           (bs, fd, fd+1) bool — True = keep, False = mask out
        action_feature: cached (fd, 1, bs, out_dim)
        state_feature:  cached (fd, fd, bs, out_dim)
        returns: Normal(bs, fd)
        """
        fd = self.state_dim

        if action_feature is None:
            action_feature = self._extract_action_feature(action)   # (fd, 1, bs, out_dim)
        if state_feature is None:
            state_feature = self._extract_state_feature(state)      # (fd, fd, bs, out_dim)

        # cat along parent dim → (fd, fd+1, bs, out_dim)
        sa_feature = torch.cat([state_feature, action_feature], dim=1)

        if mask is not None:
            # mask: (bs, fd, fd+1) → permute → (fd, fd+1, bs)
            m = mask.permute(1, 2, 0)
            sa_feature = sa_feature.clone()
            sa_feature[~m] = float('-inf')

        # max-pool over parent dim
        sa_feature, _ = sa_feature.max(dim=1)                       # (fd, bs, out_dim)

        dist = self._predict_from_sa_feature(sa_feature, state)
        return dist, action_feature, state_feature

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------

    def _nll(self, dist, target):
        """
        dist:   Normal(bs, fd)
        target: (bs, fd)
        returns scalar
        """
        return -dist.log_prob(target).mean()

    # ------------------------------------------------------------------
    # training step
    # ------------------------------------------------------------------

    def train_step(self, s_batch, a_batch):
        """
        s_batch: (bs, 2, state_dim)  — s_batch[:,0] = s_t, s_batch[:,1] = s_{t+1}
        a_batch: (bs, action_dim)
        """
        s_t   = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs    = s_t.shape[0]
        fd    = self.state_dim

        self.opt.zero_grad()

        # --- full prediction ---
        full_dist, action_feature, state_feature = self._forward(s_t, a_batch)
        full_loss = self._nll(full_dist, s_tp1)

        # --- masked prediction (drop one random parent per child) ---
        # sample one index to mask per child variable
        drop_idx = torch.randint(fd + 1, (bs, fd), device=self.device)  # (bs, fd)
        mask = self._ids_to_mask(drop_idx)                               # (bs, fd, fd+1)

        masked_dist, _, _ = self._forward(s_t, a_batch, mask=mask,
                                           action_feature=action_feature,
                                           state_feature=state_feature)
        masked_loss = self._nll(masked_dist, s_tp1)

        loss = full_loss + masked_loss
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for pl in [self.action_feature_weights, self.action_feature_biases,
                           self.state_feature_weights,  self.state_feature_biases,
                           self.generative_weights,     self.generative_biases]
               for p in pl],
            self.grad_clip
        )
        self.opt.step()
        return loss.item()

    # ------------------------------------------------------------------
    # mask update  (mirrors update_mask in original)
    # ------------------------------------------------------------------

    def update_mask(self, s_batch, a_batch):
        """
        s_batch: (bs, 2, state_dim)
        a_batch: (bs, action_dim)
        Accumulates CMI estimates; performs EMA update every eval_steps calls.
        """
        s_t   = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs    = s_t.shape[0]
        fd    = self.state_dim

        with torch.no_grad():
            # cache full features once
            action_feature = self._extract_action_feature(a_batch)
            state_feature  = self._extract_state_feature(s_t)

            full_dist, _, _ = self._forward(s_t, a_batch,
                                             action_feature=action_feature,
                                             state_feature=state_feature)
            # per-sample, per-variable NLL: (bs, fd)
            full_nll = -full_dist.log_prob(s_tp1)               # (bs, fd)

            step_cmi = torch.zeros(fd, fd + 1, device=self.device)

            for j in range(fd + 1):
                # drop parent j for every child
                drop_idx = torch.full((bs, fd), j, dtype=torch.long, device=self.device)
                # ensure diagonal (child == j) is preserved per original:
                # "each state variable must depend on itself"
                # but we allow learning here — so just mask j uniformly
                mask = self._ids_to_mask(drop_idx)               # (bs, fd, fd+1)

                masked_dist, _, _ = self._forward(s_t, a_batch, mask=mask,
                                                   action_feature=action_feature,
                                                   state_feature=state_feature)
                masked_nll = -masked_dist.log_prob(s_tp1)        # (bs, fd)

                # CMI[:,j]: how much does dropping j hurt each child
                cmi_j = (masked_nll - full_nll).mean(dim=0)      # (fd,)
                step_cmi[:, j] = cmi_j

        self._eval_cmi_acc   += step_cmi
        self._eval_step_count += 1

        if self._eval_step_count >= self.eval_steps:
            avg_cmi = self._eval_cmi_acc / self.eval_steps
            self.mask_CMI = self.eval_tau * self.mask_CMI + (1 - self.eval_tau) * avg_cmi
            self.mask = self.mask_CMI >= self.cmi_threshold

            # reset accumulators
            self._eval_cmi_acc    = torch.zeros(fd, fd + 1, device=self.device)
            self._eval_step_count = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _ids_to_mask(self, drop_idx):
        """
        drop_idx: (bs, fd) — index of parent to drop for each (sample, child) pair
        returns:  (bs, fd, fd+1) bool — True = keep
        """
        fd1   = self.state_dim + 1
        onehot = F.one_hot(drop_idx, fd1).bool()    # (bs, fd, fd+1)
        return ~onehot                               # True = keep

    def _forward_net_3d(self, x, weights, biases, final_activation=False):
        """
        Batched linear layers.
        x:       (N, bs, in_dim)
        weights: ParameterList of (N, in_dim, out_dim)
        biases:  ParameterList of (N, 1,      out_dim)
        """
        for i, (w, b) in enumerate(zip(weights, biases)):
            # x: (N, bs, in_dim), w: (N, in_dim, out_dim)
            x = torch.bmm(x, w) + b                # (N, bs, out_dim)
            is_last = (i == len(weights) - 1)
            if not is_last or final_activation:
                x = F.relu(x)
        return x

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_causal_graph(self):
        return self.mask_CMI

    def get_binary_graph(self):
        graph = (self.mask_CMI >= self.cmi_threshold).int()
        # state variables don't need self-edges in adjacency
        fd = self.state_dim
        graph[:fd, :fd].fill_diagonal_(0)
        return graph

    def parameters(self):
        for pl in [self.action_feature_weights, self.action_feature_biases,
                   self.state_feature_weights,  self.state_feature_biases,
                   self.generative_weights,     self.generative_biases]:
            for p in pl:
                yield p