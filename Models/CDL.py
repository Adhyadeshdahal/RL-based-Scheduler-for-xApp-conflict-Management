import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from Models.base import CausalModel


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

        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, feature_dim, []) for _ in range(state_dim)]
        )
        self.action_feature_extractor = MLP(action_dim, feature_dim, [])
        self.predictor = MLP(feature_dim, 2, pred_hidden)

    def forward(self, s, a, mask=None):
        """
        s:    (bs, state_dim)
        a:    (bs, action_dim)
        mask: (bs, state_dim+1) bool
        returns: mu (bs, 1), std (bs, 1)
        """
        bs = s.shape[0]
        fd = self.state_dim

        feats = []
        for i in range(fd):
            feats.append(self.state_feature_extractors[i](s[:, i : i + 1]))

        feats.append(self.action_feature_extractor(a))
        feats = torch.stack(feats, dim=1)  # (bs, fd+1, feature_dim)

        if mask is not None:
            feats = feats.masked_fill(mask.unsqueeze(-1), float("-inf"))

        h, _ = feats.max(dim=1)  # (bs, feature_dim)

        out = self.predictor(h)
        mu = out[:, 0:1]
        log_std = out[:, 1:2]
        std = torch.exp(torch.clamp(log_std, -5, 2)) + 1e-4

        return mu, std


class CDL(CausalModel):
    def __init__(
        self,
        state_dim,
        action_dim,
        kpi_start,
        feature_fc_dims,
        generative_fc_dims,
        lr,
        cmi_threshold,
        eval_tau,
        grad_clip,
        device,
        node_names,
        eval_steps=10,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cmi_threshold = cmi_threshold
        self.eval_tau = eval_tau
        self.eval_steps = eval_steps
        self.grad_clip = grad_clip
        self.kpi_start = kpi_start
        self.node_names = node_names

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        feature_dim = feature_fc_dims[-1]

        self.models = nn.ModuleList(
            [
                StatePredictor(state_dim, action_dim, feature_dim, list(generative_fc_dims))
                for _ in range(state_dim)
            ]
        ).to(self.device)

        self.opt = optim.Adam(self.models.parameters(), lr=lr)

        fd = state_dim
        self.mask_CMI = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_cmi_acc = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_step_count = 0

    def _nll(self, mu, std, target):
        return -Normal(mu, std).log_prob(target)

    def train_step(self, s_batch, a_batch):
        s_t = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs = s_t.shape[0]
        fd = self.state_dim

        self.opt.zero_grad()
        total_loss = 0.0

        # a_batch[:, 0] is param_id — the param that was changed
        # bias: drop the changed param more often so model learns to predict without it
        changed_param_ids = a_batch[:, 0].long()  # (bs,) which param changed

        # 50% of the time drop the changed param, 50% drop random
        use_informed = torch.rand(bs, device=self.device) > 0.5
        random_drop = torch.randint(fd + 1, (bs,), device=self.device)
        informed_drop = changed_param_ids  # drop the changed param

        drop_idx = torch.where(use_informed, informed_drop, random_drop)
        mask = F.one_hot(drop_idx, fd + 1).bool()

        for j in range(fd):
            target = s_tp1[:, j : j + 1]
            model = self.models[j]
            model.train()

            mu, std = model(s_t, a_batch)
            full_loss = self._nll(mu, std, target).mean()

            mu_m, std_m = model(s_t, a_batch, mask=mask)
            masked_loss = self._nll(mu_m, std_m, target).mean()

            total_loss += full_loss + masked_loss

        loss = total_loss / fd
        loss.backward()
        nn.utils.clip_grad_norm_(self.models.parameters(), self.grad_clip)
        self.opt.step()

        return loss.item()

    def update_mask(self, s_batch, a_batch):
        s_t = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs = s_t.shape[0]
        fd = self.state_dim

        step_cmi = torch.zeros(fd, fd + 1, device=self.device)

        with torch.no_grad():
            for j in range(fd):
                target = s_tp1[:, j : j + 1]
                self.models[j].eval()
                mu, std = self.models[j](s_t, a_batch)
                full_nll = self._nll(mu, std, target)

                for i in range(fd + 1):
                    mask = torch.zeros(bs, fd + 1, dtype=torch.bool, device=self.device)
                    self.models[j].eval()
                    mask[:, i] = True
                    mu_m, std_m = self.models[j](s_t, a_batch, mask=mask)
                    masked_nll = self._nll(mu_m, std_m, target)
                    step_cmi[j, i] = (masked_nll - full_nll).mean()

        self._eval_cmi_acc += step_cmi
        self._eval_step_count += 1

        if self._eval_step_count >= self.eval_steps:
            avg_cmi = self._eval_cmi_acc / self.eval_steps
            self.mask_CMI = self.eval_tau * self.mask_CMI + (1 - self.eval_tau) * avg_cmi
            self._eval_cmi_acc = torch.zeros(fd, fd + 1, device=self.device)
            self._eval_step_count = 0

    def get_causal_graph(self):
        return self.mask_CMI

    def get_binary_graph(self, threshold=None):
        if threshold is None:
            threshold = self.cmi_threshold
        graph = self.mask_CMI >= threshold
        fd = graph.shape[0]
        if graph.shape[1] > fd:
            graph[:fd, :fd].fill_diagonal_(0)
        return graph

    def predictNextState(self, s, a):
        bs = s.shape[0]
        s = s.to(self.device)
        a = a.to(self.device)
        mus, stds = [], []
        with torch.no_grad():
            for j in range(self.kpi_start, self.state_dim):
                self.models[j].eval()
                graph_mask = self.get_binary_graph()[j, :].clone()  # (fd+1,)
                graph_mask[j] = True  # always include self
                graph_mask[-1] = True  # always include action
                mask = graph_mask.unsqueeze(0).expand(bs, -1).bool().to(self.device)
                mu, std = self.models[j](s, a, ~mask)
                mus.append(mu)
                stds.append(std)
        mu = torch.cat(mus, dim=1)
        std = torch.cat(stds, dim=1)
        return Normal(mu, std)

    def evaluatePredictions(self, s, a, s_1):
        """
        s:   (bs, state_dim)
        a:   (bs, 1)
        s_1: (bs, state_dim)
        returns: scalar MSE
        """
        dist = self.predictNextState(s, a)
        pred = dist.sample()  # (bs, state_dim - kpi_start)
        target = s_1[:, self.kpi_start :]  # (bs, state_dim - kpi_start)
        return ((pred - target) ** 2).mean().item()  # scalar

    def save_model(self, filepath="cdl_model.pt"):
        """
        Save the model state, optimizer state, and other necessary attributes.
        """
        state = {
            "models_state_dict": [model.state_dict() for model in self.models],
            "optimizer_state_dict": self.opt.state_dict(),
            "mask_CMI": self.mask_CMI,
            "eval_cmi_acc": self._eval_cmi_acc,
            "eval_step_count": self._eval_step_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cmi_threshold": self.cmi_threshold,
            "eval_tau": self.eval_tau,
            "eval_steps": self.eval_steps,
            "grad_clip": self.grad_clip,
            "device": self.device,
        }
        torch.save(state, filepath)

    def load_model(self, filepath="cdl_model.pt"):
        """
        Load the model state, optimizer state, and other attributes.
        """
        state = torch.load(filepath, map_location=self.device)
        for i, model in enumerate(self.models):
            model.load_state_dict(state["models_state_dict"][i])
        self.opt.load_state_dict(state["optimizer_state_dict"])
        self.mask_CMI = state["mask_CMI"]
        self._eval_cmi_acc = state["eval_cmi_acc"]
        self._eval_step_count = state["eval_step_count"]
