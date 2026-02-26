import torch
import torch.nn as nn
import torch.optim as optim
import random


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
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, hidden_dim, []) for _ in range(state_dim)]
        )
        self.action_feature_extractor = MLP(action_dim, hidden_dim, [])
        self.predictor = MLP(hidden_dim, 2, pred_hidden)

    def forward(self, s, a, mask=None):

        feats = [self.action_feature_extractor(a)]  

        for i in range(self.state_dim):
            fi = self.state_feature_extractors[i](s[:, i:i+1])
            if mask is not None and mask[i] == 0:
                fi = torch.full_like(fi, -1e9)
            feats.append(fi)

        h   = torch.stack(feats, dim=0).max(dim=0).values
        out = self.predictor(h)
        mean = out[:, 0:1]
        std  = torch.clamp(torch.exp(out[:, 1:2]), 1e-3, 1e-2)
        return mean, std


class CDL:

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim    = 64,
        pred_hidden   = [64, 32],
        lr            = 3e-4,
        cmi_threshold = 0.01,   # relaxed — use heatmap to tune for your env
        ema_decay     = 0.999,
        device        = None,
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

        # cmi_matrix[i, j] = normalized CMI(s^i_t -> s^j_{t+1})
        self.cmi_matrix = torch.zeros(state_dim, state_dim, device=self.device)



    def gaussian_nll(self, mean, std, target):
        var = std ** 2
        return 0.5 * torch.log(2 * torch.pi * var) + (target - mean) ** 2 / (2 * var)

    def _parent_mask(self, j):
        """Binary mask keeping only CMI-inferred parents of s^j."""
        pa_mask = (self.cmi_matrix[:, j] > self.cmi_threshold).float()
        return pa_mask



    def full_cdl_loss(self, s_batch, a_batch):

        s      = s_batch[:, 0]
        s_next = s_batch[:, 1]
        a      = a_batch

        loss = torch.tensor(0.0, device=self.device)

        for j in range(self.state_dim):
            target = s_next[:, j:j+1]

            # Term 1: full input
            mean, std = self.models[j](s, a, mask=None)
            loss = loss + self.gaussian_nll(mean, std, target).mean()

            # Term 2: mask out a random i != j
            i = random.choice([k for k in range(self.state_dim) if k != j])
            mask_i    = torch.ones(self.state_dim, device=self.device)
            mask_i[i] = 0
            m_mean, m_std = self.models[j](s, a, mask=mask_i)
            loss = loss + self.gaussian_nll(m_mean, m_std, target).mean()

            # Term 3: parent-only (skip early when graph is empty)
            if self.cmi_matrix.max().item() > self.cmi_threshold:
                pa_mask = self._parent_mask(j)
                p_mean, p_std = self.models[j](s, a, mask=pa_mask)
                loss = loss + self.gaussian_nll(p_mean, p_std, target).mean()

        return loss

    def train_step(self, s_batch, a_batch):
        self.opt.zero_grad()
        loss = self.full_cdl_loss(s_batch, a_batch)
        loss.backward()
        self.opt.step()
        return loss.item()



    def evaluate_cmi(self, s_val, a_val):

        s      = s_val[:, 0]
        s_next = s_val[:, 1]
        a      = a_val

        with torch.no_grad():
            for j in range(self.state_dim):
                target = s_next[:, j:j+1]

                full_mean, full_std = self.models[j](s, a, mask=None)
                full_nll = self.gaussian_nll(full_mean, full_std, target).mean()

                for i in range(self.state_dim):
                    if i == j:
                        continue  

                    mask_i    = torch.ones(self.state_dim, device=self.device)
                    mask_i[i] = 0
                    m_mean, m_std = self.models[j](s, a, mask=mask_i)
                    masked_nll = self.gaussian_nll(m_mean, m_std, target).mean()

                    raw_cmi  = torch.clamp(masked_nll - full_nll, min=0.0)
                    norm_cmi = raw_cmi / (full_nll.abs() + 1e-8)

                    self.cmi_matrix[i, j] = (
                        self.ema_decay       * self.cmi_matrix[i, j]
                        + (1 - self.ema_decay) * norm_cmi.item()
                    )



    def get_causal_graph(self):
        graph = (self.cmi_matrix > self.cmi_threshold).int()
        graph.fill_diagonal_(1)
        return graph

    def get_cmi_matrix(self):
        return self.cmi_matrix

    def to(self, device):
        self.device     = torch.device(device)
        self.models     = self.models.to(self.device)
        self.cmi_matrix = self.cmi_matrix.to(self.device)
        return self