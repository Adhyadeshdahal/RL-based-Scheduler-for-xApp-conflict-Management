import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from Models.base import WorldModel

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=(64, 32)):
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


class MLPInference(WorldModel):
    def __init__(
        self,
        state_dim,
        action_dim,
        kpi_start,
        generative_fc_dims,
        feature_fc_dims,
        lr,
        cmi_threshold,
        eval_tau,
        grad_clip,
        device,
        node_names,
        hidden_layers=(64, 32),
        eval_steps=10,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.kpi_start = kpi_start
        self.eval_steps = eval_steps
        self.grad_clip = grad_clip
        self.hidden_layers = hidden_layers

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = self.models = MLP(
            input_dim=state_dim + action_dim,
            output_dim=2 * state_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def _nll(self, mu, std, target):
        return -Normal(mu, std).log_prob(target)

    def _forward(self, s, a):
        """
        s: (bs, state_dim)
        a: (bs, action_dim)
        """
        x = torch.cat([s, a], dim=-1)  # (bs, state_dim + action_dim)
        out = self.model(x)  # (bs, 2 * state_dim)
        mu, log_std = out.chunk(2, dim=-1)  # each (bs, state_dim)
        std = F.softplus(log_std) + 1e-6
        return mu, std

    def train_step(self, s_batch, a_batch):
        """
        s_batch: (bs, 2, state_dim)
        a_batch: (bs, action_dim)
        """
        s_t = s_batch[:, 0]  # (bs, state_dim)
        s_tp1 = s_batch[:, 1]  # (bs, state_dim)

        self.model.train()
        self.opt.zero_grad()

        mu, std = self._forward(s_t, a_batch)
        loss = self._nll(mu, std, s_tp1).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()

        return loss.item()

    def predictNextState(self, s, a):
        """
        s: (bs, state_dim)
        a: (bs, action_dim)
        returns: Normal with mean/scale (bs, state_dim - kpi_start)
        """
        s = s.to(self.device)
        a = a.to(self.device)
        self.model.eval()
        with torch.no_grad():
            mu, std = self._forward(s, a)  # (bs, state_dim)
        mu = mu[:, self.kpi_start :]  # (bs, n_kpis)
        std = std[:, self.kpi_start :]  # (bs, n_kpis)
        return Normal(mu, std)

    def evaluatePredictions(self, s, a, s_1):
        """
        s:   (bs, state_dim)
        a:   (bs, action_dim)
        s_1: (bs, state_dim)
        returns: scalar MSE
        """
        dist = self.predictNextState(s, a)
        pred = dist.sample()  # (bs, n_kpis)
        target = s_1[:, self.kpi_start :]  # (bs, n_kpis)
        return ((pred - target) ** 2).mean().item()

    def save_model(self, filepath):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "kpi_start": self.kpi_start,
                "hidden_layers": self.hidden_layers,
                "eval_steps": self.eval_steps,
                "grad_clip": self.grad_clip,
            },
            filepath,
        )
        logger.info("Model saved to %s", filepath)

    def load_model(self, filepath):
        state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.opt.load_state_dict(state["optimizer_state_dict"])
        logger.info("Model loaded from %s", filepath)
