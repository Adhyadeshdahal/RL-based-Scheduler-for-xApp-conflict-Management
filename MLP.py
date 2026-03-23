import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


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


class MLPInference:

    def __init__(
        self,
        state_dim,
        action_dim,
        generative_fc_dims = (64,32),
        feature_fc_dims=(64,),
        hidden_layers=(64, 32),
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
        self.hidden_layers = hidden_layers

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Input: state_dim (s_t + a combined), Output: 2 * state_dim (mu + log_std)
        self.model =self.models= MLP(
            input_dim=state_dim,
            output_dim=2 * state_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        self.mask_CMI         = torch.zeros(state_dim, state_dim + 1, device=self.device)
        self._eval_cmi_acc    = torch.zeros(state_dim, state_dim + 1, device=self.device)
        self._eval_step_count = 0


    def _nll(self, mu, std, target):
        """Negative log-likelihood under a diagonal Gaussian."""
        return -Normal(mu, std).log_prob(target)

    def _forward(self, x):
        """Run model and return (mu, std)."""
        out = self.model(x)                          # (bs, 2 * state_dim)
        mu, log_std = out.chunk(2, dim=-1)           # each (bs, state_dim)
        std = F.softplus(log_std) + 1e-6             # positive std, no blow-up
        return mu, std



    def train_step(self, s_batch, a_batch):
        """
        One gradient update on a batch.

        s_batch : (bs, 2, state_dim)  — s_batch[:,0] = s_t, s_batch[:,1] = s_{t+1}
        a_batch : (bs, state_dim)     — action (same dim as state for additive mix)

        Returns scalar loss value.
        """
        s_t   = s_batch[:, 0] + a_batch   # (bs, state_dim)
        s_tp1 = s_batch[:, 1]             # (bs, state_dim)  — regression target

        self.model.train()
        self.opt.zero_grad()

        mu, std = self._forward(s_t)
        loss    = self._nll(mu, std, s_tp1).mean()   # scalar

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()

        return loss.item()



    def update_mask(self, s_batch, a_batch):
        pass

    def get_causal_graph(self):
        pass

    def get_binary_graph(self):
        pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predictNextState(self, s, a):
        kpi_start = 8
        self.model.eval()
        with torch.no_grad():
            mu, std = self._forward(s + a)           # (bs, state_dim)

        dists = [
            Normal(mu[:, i:i+1], std[:, i:i+1])
            for i in range(kpi_start, self.state_dim)
        ]
        return dists

    def save_model(self, filepath="mlp_model.pt"):
        """Save model weights, optimizer state, and hyperparameters."""
        torch.save(
            {
                # Weights & optimiser
                "model_state_dict"    : self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                # Runtime state
                "mask_CMI"            : self.mask_CMI,
                "eval_cmi_acc"        : self._eval_cmi_acc,
                "eval_step_count"     : self._eval_step_count,
                # Hyperparameters (needed to reconstruct the object)
                "state_dim"           : self.state_dim,
                "action_dim"          : self.action_dim,
                "hidden_layers"       : self.hidden_layers,
                "cmi_threshold"       : self.cmi_threshold,
                "eval_tau"            : self.eval_tau,
                "eval_steps"          : self.eval_steps,
                "grad_clip"           : self.grad_clip,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="mlp_model.pt"):
        """Load model weights, optimizer state, and runtime state."""
        state = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.opt.load_state_dict(state["optimizer_state_dict"])

        self.mask_CMI         = state["mask_CMI"]
        self._eval_cmi_acc    = state["eval_cmi_acc"]
        self._eval_step_count = state["eval_step_count"]
        print(f"Model loaded from {filepath}")

