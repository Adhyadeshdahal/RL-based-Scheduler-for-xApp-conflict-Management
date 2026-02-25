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
    """
    Predicts state variable s^j_{t+1} from (a_t, s_t).

    Architecture (per paper Sec 3.3 / Fig 3):
      1. Each state variable s^i_t and the action a_t are individually
         mapped to a hidden_dim-dimensional feature vector.
      2. Selected features are masked to -inf according to mask M^j.
      3. Element-wise max over all features gives h^j.
      4. A predictive MLP q^j maps h^j -> distribution over s^j_{t+1}.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, pred_hidden):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # One feature extractor per state variable + one for the action
        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, hidden_dim, []) for _ in range(state_dim)]
        )
        self.action_feature_extractor = MLP(action_dim, hidden_dim, [])

        # Predictive network: h^j -> (mean, log_std)
        self.predictor = MLP(hidden_dim, 2, pred_hidden)

    def forward(self, s, a, mask=None):
        """
        Args:
            s    : (B, state_dim)  current state
            a    : (B, action_dim) current action
            mask : (state_dim,) binary mask over state features.
                   mask[i] = 0 means s^i is masked out (set to -inf).
                   Action is never masked (always in the conditioning set
                   unless you pass action_masked=True).
        Returns:
            mean : (B, 1)
            std  : (B, 1)  clamped to [1e-3, 1e-2]
        """
        feats = []

        # Action feature (never masked)
        fa = self.action_feature_extractor(a)           # (B, hidden_dim)
        feats.append(fa)

        # State features (optionally masked)
        for i in range(self.state_dim):
            fi = self.state_feature_extractors[i](s[:, i:i+1])  # (B, hidden_dim)
            if mask is not None and mask[i] == 0:
                fi = torch.full_like(fi, -1e9)
            feats.append(fi)

        # Element-wise max pooling  (paper eq. before Eq.3)
        h = torch.stack(feats, dim=0).max(dim=0).values  # (B, hidden_dim)

        out = self.predictor(h)
        mean = out[:, 0:1]
        std  = torch.clamp(torch.exp(out[:, 1:2]), 1e-3, 1e-2)
        return mean, std


class CDL:
    """
    Causal Dynamics Learning (CDL) — Chang et al., ICML 2022.

    Learns a causal dynamics model p(s_{t+1} | a_t, s_t) by:
      1. Training dS predictive models with the three-term loss (Eq. 3).
      2. Estimating CMI_{ij} on held-out data to discover causal edges.
      3. Exposing the learned causal graph via get_causal_graph().
    """
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

        # Auto-select GPU if available, unless device is explicitly specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # One StatePredictor per target variable s^j_{t+1}
        self.models = nn.ModuleList([
            StatePredictor(state_dim, action_dim, hidden_dim, pred_hidden)
            for _ in range(state_dim)
        ]).to(self.device)
        self.opt = optim.Adam(self.models.parameters(), lr=lr)

        # CMI matrix: cmi_matrix[i, j] ≈ CMI(s^i_t; s^j_{t+1} | {a_t, s_t \ s^i_t})
        # Positive value => causal edge s^i_t -> s^j_{t+1} likely exists.
        self.cmi_matrix = torch.zeros(state_dim, state_dim, device=self.device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Training  (Eq. 3 — single-step transitions)
    # ------------------------------------------------------------------

    def full_cdl_loss(self, s_batch, a_batch):
        """
        Computes the CDL training loss (Eq. 3) on a batch of
        single-step transitions.

        Args:
            s_batch : (B, 2, state_dim)  s_batch[:, 0] = s_t,
                                          s_batch[:, 1] = s_{t+1}
            a_batch : (B, action_dim)    a_t

        Returns:
            Scalar loss (sum over j of three NLL terms).
        """
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
        Estimates NORMALIZED CMI_{ij} for all (i, j) pairs on validation data
        and updates self.cmi_matrix via EMA.

        Raw CMI = NLL_masked - NLL_full
        Problem: indirect correlations (e.g. P1->K1->K3 making P1 appear as
        a direct cause of K3) produce non-zero raw CMI because the model
        has learned to exploit them. This causes many false positive edges.

        Fix — normalize by absolute full NLL:
            nCMI_{ij} = (NLL_masked - NLL_full) / (|NLL_full| + eps)

        This expresses the RELATIVE prediction degradation when s^i is removed.
        Direct causes cause a large relative drop. Indirect correlations cause
        a small absolute CMI relative to baseline difficulty, so they normalize
        away and fall below the threshold.

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
                    if i == j:
                        # Skip self-edges — they are forced in get_causal_graph()
                        # per Assumption A3. Params are reset randomly each step
                        # so their learned self-CMI is ~0, but A3 still requires
                        # self-edges for all variables.
                        continue

                    mask_i = torch.ones(self.state_dim, device=self.device)
                    mask_i[i] = 0
                    m_mean, m_std = self.models[j](s, a, mask=mask_i)
                    masked_nll = self.gaussian_nll(m_mean, m_std, target).mean()

                    # Raw CMI (clamp negatives — model noise)
                    raw_cmi = torch.clamp(masked_nll - full_nll, min=0.0)

                    # Normalize: suppresses weak indirect correlations
                    norm_cmi = raw_cmi / (full_nll.abs() + 1e-8)

                    # EMA update
                    self.cmi_matrix[i, j] = (
                        self.ema_decay       * self.cmi_matrix[i, j]
                        + (1 - self.ema_decay) * norm_cmi.item()
                    )

    # ------------------------------------------------------------------
    # Causal graph
    # ------------------------------------------------------------------

    def get_causal_graph(self, force_self_edges: bool = True):
        """
        Returns a (state_dim x state_dim) binary adjacency matrix where
        entry [i, j] = 1 iff the causal edge s^i_t -> s^j_{t+1} is inferred.

        Args:
            force_self_edges : if True, diagonal is set to 1 per Assumption A3.
                               Necessary because param self-CMI is ~0 (params
                               are randomly reset each step) even though A3
                               requires s^i_t -> s^i_{t+1} for all i.
        """
        graph = (self.cmi_matrix > self.cmi_threshold).int()
        if force_self_edges:
            graph.fill_diagonal_(1)
        return graph

    def get_cmi_matrix_normalized(self):
        """Returns the normalized CMI matrix (before thresholding) for inspection."""
        return self.cmi_matrix.clone()

    # ------------------------------------------------------------------
    # State Abstraction  (Definitions 1-3, Sec 3.2)
    # ------------------------------------------------------------------

    def get_state_abstraction(self):
        """
        Derives the state abstraction φ(s) = (s^C, s^R) by classifying
        each state variable into one of three types from the causal graph:

          Controllable (sC)       — Definition 1: descendants of the action.
                                    i.e., action has a causal edge into s^j
                                    (action_col of cmi_matrix > threshold).

          Action-Relevant (sR)    — Definition 2: ancestors of sC that are
                                    not themselves in sC. These variables
                                    causally influence sC even though the
                                    action does not directly control them.

          Action-Irrelevant (sI)  — Definition 3: everything else. These are
                                    pruned from the abstract dynamics.

        The method also stores action_cmi — the CMI between the action and
        each s^j — which must be populated via evaluate_action_cmi() before
        calling this method.

        Returns:
            abstraction : dict with keys
                "sC"    : sorted list of controllable variable indices
                "sR"    : sorted list of action-relevant variable indices
                "sI"    : sorted list of action-irrelevant variable indices
                "graph" : (state_dim x state_dim) binary adjacency tensor
        """
        graph = self.get_causal_graph().cpu()  # (D, D)  graph[i,j]=1 => s^i -> s^j

        # --- Step 1: Controllable (sC) — direct descendants of action --------
        # action_cmi[j] > threshold  =>  action -> s^j  exists
        if not hasattr(self, "action_cmi"):
            raise RuntimeError(
                "action_cmi not found. Call evaluate_action_cmi() first."
            )
        action_cmi_cpu = self.action_cmi.cpu()
        sC = set(
            int(j) for j in (action_cmi_cpu > self.cmi_threshold).nonzero(as_tuple=True)[0]
        )

        # --- Step 2: Action-Relevant (sR) — ancestors of sC, not in sC -------
        # Traverse the graph backwards from sC to find all ancestors.
        sR = set()
        frontier = set(sC)
        while frontier:
            new_frontier = set()
            for j in frontier:
                # any i with graph[i, j] == 1 is a parent of j
                parents = set(
                    int(i) for i in graph[:, j].nonzero(as_tuple=True)[0]
                )
                # only add parents that are not already classified
                new_parents = parents - sC - sR
                sR.update(new_parents)
                new_frontier.update(new_parents)
            frontier = new_frontier

        # --- Step 3: Action-Irrelevant (sI) — everything else ----------------
        all_vars = set(range(self.state_dim))
        sI = all_vars - sC - sR

        abstraction = {
            "sC":    sorted(sC),
            "sR":    sorted(sR),
            "sI":    sorted(sI),
            "graph": graph,
        }
        self._abstraction = abstraction
        return abstraction

    def evaluate_action_cmi(self, s_val, a_val):
        """
        Estimates CMI between the action a_t and each s^j_{t+1}
        (i.e., tests at ⊥ s^j_{t+1} | s_t per Theorem 3.1).

        Concretely:
            action_cmi[j] = E[ NLL(s^j | s_t, zeros_action)
                               - NLL(s^j | s_t, a_t) ]

        We approximate "masking the action" by passing a zero action,
        which zeroes out the action's contribution via the feature extractor.

        Args:
            s_val : (B, 2, state_dim)
            a_val : (B, action_dim)
        """
        s      = s_val[:, 0]
        s_next = s_val[:, 1]
        a      = a_val
        a_zero = torch.zeros_like(a)  # action-masked baseline

        if not hasattr(self, "action_cmi"):
            self.action_cmi = torch.zeros(self.state_dim, device=self.device)

        with torch.no_grad():
            for j in range(self.state_dim):
                target = s_next[:, j:j+1]

                # Full NLL: conditioned on (s_t, a_t)
                full_mean, full_std = self.models[j](s, a, mask=None)
                full_nll = self.gaussian_nll(full_mean, full_std, target).mean()

                # Action-masked NLL: conditioned on (s_t, 0)
                masked_mean, masked_std = self.models[j](s, a_zero, mask=None)
                masked_nll = self.gaussian_nll(masked_mean, masked_std, target).mean()

                cmi = torch.clamp(masked_nll - full_nll, min=0.0)

                # EMA update
                self.action_cmi[j] = (
                    self.ema_decay * self.action_cmi[j]
                    + (1 - self.ema_decay) * cmi.item()
                )

    def get_abstract_state(self, s):
        """
        Applies the abstraction φ(s) = (s^C, s^R), dropping s^I.

        Args:
            s : (B, state_dim) tensor  or  (state_dim,) tensor
        Returns:
            s_abstract : (B, |sC| + |sR|) tensor
            index_map  : list of original indices kept  [sC indices, sR indices]
        """
        if not hasattr(self, "_abstraction"):
            raise RuntimeError("Call get_state_abstraction() first.")
        keep = self._abstraction["sC"] + self._abstraction["sR"]
        return s[..., keep], keep

    def predict_abstract(self, s_abstract, a, index_map):
        """
        Runs the causal dynamics model Fφ_θ in the abstract space,
        i.e., only for sC and sR variables, skipping sI predictors entirely.

        Args:
            s_abstract : (B, |sC|+|sR|) tensor — abstract state φ(s_t)
            a          : (B, action_dim) tensor
            index_map  : list of original variable indices (from get_abstract_state)
        Returns:
            means : (B, |sC|+|sR|)
            stds  : (B, |sC|+|sR|)
        """
        if not hasattr(self, "_abstraction"):
            raise RuntimeError("Call get_state_abstraction() first.")

        # Build a full-dim state tensor filled with zeros, insert abstract dims
        B = s_abstract.shape[0]
        s_full = torch.zeros(B, self.state_dim, device=self.device)
        idx = torch.tensor(index_map, device=self.device)
        s_full[:, idx] = s_abstract

        means, stds = [], []
        for j in index_map:
            pa_mask = self._parent_mask(j)
            mean, std = self.models[j](s_full, a, mask=pa_mask)
            means.append(mean)
            stds.append(std)

        return torch.cat(means, dim=1), torch.cat(stds, dim=1)

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
#   cdl = CDL(state_dim=11, action_dim=7)
#   print(cdl.device)   # cuda:0 if GPU is available, else cpu
#
#   # Move data to device before passing in
#   s_batch = s_batch.to(cdl.device)   # (B, 2, state_dim)
#   a_batch = a_batch.to(cdl.device)   # (B, action_dim)
#
#   # --- Training ---
#   for step in range(num_steps):
#       loss = cdl.train_step(s_batch, a_batch)
#       if step % 500 == 0:
#           cdl.evaluate_cmi(s_val, a_val)
#           cdl.evaluate_action_cmi(s_val, a_val)   # needed for abstraction
#
#   # --- State Abstraction (Definitions 1-3) ---
#   abstraction = cdl.get_state_abstraction()
#   print("Controllable   (sC):", abstraction["sC"])    # direct action descendants
#   print("Action-Relevant(sR):", abstraction["sR"])    # ancestors of sC
#   print("Action-Irrelevant(sI):", abstraction["sI"])  # pruned from dynamics
#
#   # --- Abstract-space rollout (Fph_th) ---
#   s_abstract, index_map = cdl.get_abstract_state(s_t)
#   means, stds = cdl.predict_abstract(s_abstract, a_t, index_map)