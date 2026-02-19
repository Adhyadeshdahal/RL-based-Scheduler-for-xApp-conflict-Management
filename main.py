"""
Causal RL at Pearl's Level 2 (Interventional).

Architecture:
  Environment M  = SCM  — KPI physics are structural equations, graph is causal DAG
  Agent G        = causal belief network — HGT loaded from trained weights,
                   fine-tuned via PPO to learn interventional policy do(param=x)

Why L2 and not L1/L3:
  L1 (association): HGT KPI predictor — observes P(KPI | graph state)
  L2 (intervention): THIS FILE — agent does do(TXP += delta), observes causal response
  L3 (counterfactual): would require explicit SCM inversion, not implemented

The causal gate explicitly separates:
  - delta_kpi features  → caused by the agent's last intervention (L2 signal)
  - abs_kpi features    → baseline state, may be spuriously correlated (L1 signal)
  The gate learns to upweight L2 signal for the actor, and use both for critic.
"""

from __future__ import annotations
import math
import os

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData


def _cco_throughput(txp, ret):
    txp_n = (txp - 6) / 34
    base = 1000 / (1 + math.exp(-10 * (txp_n - 0.5)))
    return max(10.0, base - abs(ret - 3.0) * 30)


def _cco_sinr(txp, ret):
    return max(-10.0, min(30.0, -10 + (txp - 6) * 1.2 - abs(ret - 3.0) * 0.5))


def _es_power(txp):
    return max(100.0, min(1500.0, 100 + 10 * (txp - 6) + 0.8 * (txp - 6) ** 2))


def _es_efficiency(txp):
    return max(0.1, min(10.0, 10 * math.exp(-((txp - 20) ** 2) / 200)))


def _mro_ho_success(cio, ttt, hys, txp):
    s = (
        0.3 * math.exp(-((cio - 2) ** 2) / 8)
        + 0.3 * math.exp(-((ttt - 100) ** 2) / (2 * 50**2))
        + 0.2 * math.exp(-((hys - 1.5) ** 2) / 2)
        + 0.2 * math.exp(-((txp - 20) ** 2) / 150)
    )
    return max(90.0, min(99.0, 90 + 9 * s))


def _mro_call_drop(cio, ttt, hys, txp):
    return max(
        0.0,
        min(
            2.0,
            0.2
            + abs(cio - 2) * 0.15
            + abs(ttt - 100) / 100 * 0.3
            + abs(hys - 1.5) * 0.2
            + abs(txp - 20) / 20 * 0.2,
        ),
    )


def _mro_call_block(cio, ttt, txp):
    return max(
        0.0,
        min(
            2.0,
            0.3
            + abs(ttt - 100) / 80 * 0.5
            + abs(cio - 2) * 0.1
            + max(0, (15 - txp)) / 10 * 0.3,
        ),
    )


def _mlb_traffic_load(txp, cio, ttt):
    return max(
        30.0, min(80.0, 55 - (txp - 20) * 0.5 + abs(cio) * 2 + abs(ttt - 120) / 100 * 8)
    )


def _mlb_resource_util(txp, cio, ttt):
    return max(
        20.0,
        min(70.0, 40 - abs(cio) * 1.5 - abs(ttt - 100) / 100 * 10 + (txp - 20) * 0.3),
    )


KPI_META = {
    # name                  structural equation                                    direction
    "CCO_Throughput": {"fn": lambda p: _cco_throughput(p["TXP"], p["RET"]), "dir": +1},
    "CCO_SINR": {"fn": lambda p: _cco_sinr(p["TXP"], p["RET"]), "dir": +1},
    "ES_Power": {"fn": lambda p: _es_power(p["TXP"]), "dir": -1},
    "ES_Energy_Eff": {"fn": lambda p: _es_efficiency(p["TXP"]), "dir": +1},
    "MRO_HO_Success": {
        "fn": lambda p: _mro_ho_success(p["CIO"], p["TTT"], p["HYS"], p["TXP"]),
        "dir": +1,
    },
    "MRO_Call_Drop": {
        "fn": lambda p: _mro_call_drop(p["CIO"], p["TTT"], p["HYS"], p["TXP"]),
        "dir": -1,
    },
    "MRO_Call_Block": {
        "fn": lambda p: _mro_call_block(p["CIO"], p["TTT"], p["TXP"]),
        "dir": -1,
    },
    "MLB_Traffic_Load": {
        "fn": lambda p: _mlb_traffic_load(p["TXP"], p["CIO"], p["TTT"]),
        "dir": -1,
    },
    "MLB_Resource_Util": {
        "fn": lambda p: _mlb_resource_util(p["TXP"], p["CIO"], p["TTT"]),
        "dir": +1,
    },
}
KPI_NAMES = list(KPI_META.keys())
PARAM_NAMES = ["TXP", "RET", "CIO", "TTT", "HYS"]
XAPP_NAMES = ["CCO", "ES", "MRO", "MLB"]

PARAM_BOUNDS = {
    "TXP": (6, 40),
    "RET": (0, 15),
    "CIO": (-5, 5),
    "TTT": (0, 512),
    "HYS": (0, 10),
}
PARAM_INIT = {"TXP": 20.0, "RET": 3.0, "CIO": 2.0, "TTT": 100.0, "HYS": 1.5}

# which xApp controls which params (defines the intervention structure)
XAPP_CONTROLS = {
    "CCO": ["TXP", "RET"],
    "ES": ["TXP"],
    "MRO": ["CIO", "TTT", "HYS"],
    "MLB": ["CIO", "TTT", "TXP"],
}
XAPP_OBSERVES = {
    "CCO": ["CCO_Throughput", "CCO_SINR"],
    "ES": ["ES_Power", "ES_Energy_Eff"],
    "MRO": ["MRO_HO_Success", "MRO_Call_Drop", "MRO_Call_Block"],
    "MLB": ["MLB_Traffic_Load", "MLB_Resource_Util"],
}
N_ACTIONS = sum(len(v) for v in XAPP_CONTROLS.values())  # = 9

# normalisation ranges (from simulation physics bounds)
KPI_RANGES = {
    "CCO_Throughput": (10, 1000),
    "CCO_SINR": (-10, 30),
    "ES_Power": (100, 1500),
    "ES_Energy_Eff": (0.1, 10),
    "MRO_HO_Success": (90, 99),
    "MRO_Call_Drop": (0, 2),
    "MRO_Call_Block": (0, 2),
    "MLB_Traffic_Load": (30, 80),
    "MLB_Resource_Util": (20, 70),
}

# action step sizes per param (how much one unit of action changes the param)
ACTION_SCALE = {"TXP": 2.0, "RET": 0.5, "CIO": 0.3, "TTT": 10.0, "HYS": 0.2}


HGT_HIDDEN = 128
HGT_HEADS = 4
HGT_LAYERS = 3

HGT_METADATA = (
    ["xapp", "param", "kpi"],
    [
        ("xapp", "changes", "param"),
        ("param", "influences", "kpi"),
        ("xapp", "to", "kpi"),
    ],
)


class ORANEnv(gym.Env):
    def __init__(self, max_steps: int = 300):
        super().__init__()
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_ACTIONS,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -5.0, 5.0, shape=(len(KPI_NAMES) * 2,), dtype=np.float32
        )
        self._step = 0
        self.params: Dict[str, float] = {}
        self.prev_kpis: Dict[str, float] = {}
        self.curr_kpis: Dict[str, float] = {}
        self.xapp_idx = {n: i for i, n in enumerate(XAPP_NAMES)}
        self.param_idx = {n: i for i, n in enumerate(PARAM_NAMES)}
        self.kpi_idx = {n: i for i, n in enumerate(KPI_NAMES)}
        self._build_static_edges()

    def _build_static_edges(self):
        """Pre-compute graph edges — these encode the causal DAG topology."""
        xp_src, xp_dst = [], []
        for xa, params in XAPP_CONTROLS.items():
            for p in params:
                xp_src.append(self.xapp_idx[xa])
                xp_dst.append(self.param_idx[p])
        self._xapp_param_edges = torch.tensor([xp_src, xp_dst], dtype=torch.long)

        # param→kpi edges: detect by sensitivity (which params affect which KPI)
        pk_src, pk_dst = [], []
        base_params = dict(PARAM_INIT)
        for kname, meta in KPI_META.items():
            base_val = meta["fn"](base_params)
            for pname in PARAM_NAMES:
                perturbed = dict(base_params)
                perturbed[pname] += 1.0
                if abs(meta["fn"](perturbed) - base_val) > 1e-6:
                    pk_src.append(self.param_idx[pname])
                    pk_dst.append(self.kpi_idx[kname])
        self._param_kpi_edges = torch.tensor([pk_src, pk_dst], dtype=torch.long)

        xk_src, xk_dst = [], []
        for xa, kpis in XAPP_OBSERVES.items():
            for k in kpis:
                xk_src.append(self.xapp_idx[xa])
                xk_dst.append(self.kpi_idx[k])
        self._xapp_kpi_edges = torch.tensor([xk_src, xk_dst], dtype=torch.long)

    def _norm(self, name: str, val: float) -> float:
        lo, hi = KPI_RANGES[name]
        return (val - lo) / (hi - lo + 1e-8)

    def _compute_kpis(self) -> Dict[str, float]:
        return {k: meta["fn"](self.params) for k, meta in KPI_META.items()}

    def _obs(self) -> np.ndarray:
        norm = [self._norm(k, self.curr_kpis[k]) for k in KPI_NAMES]
        delta = [
            self._norm(k, self.curr_kpis[k]) - self._norm(k, self.prev_kpis[k])
            for k in KPI_NAMES
        ]
        return np.array(norm + delta, dtype=np.float32)

    def _causal_reward(self) -> float:
        """
        L2 reward: responds only to interventional change, not to absolute KPI value.
        A policy that does nothing gets exactly 0 reward every step.
        This forces the agent to actually intervene and cause improvement.
        """
        r = 0.0
        for k in KPI_NAMES:
            lo, hi = KPI_RANGES[k]
            delta_norm = (self.curr_kpis[k] - self.prev_kpis[k]) / (hi - lo + 1e-8)
            r += KPI_META[k]["dir"] * delta_norm
        # structural penalty: high power is always bad regardless of delta
        r -= 0.05 * self._norm("ES_Power", self.curr_kpis["ES_Power"])
        return float(r)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self.params = dict(PARAM_INIT)
        self.curr_kpis = self._compute_kpis()
        self.prev_kpis = dict(self.curr_kpis)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        """
        Each action value is a do() intervention:
            do(param_i  +=  action[i] * scale_i)
        Params are clamped to physical bounds after intervention.
        """
        action = np.clip(action, -1.0, 1.0)
        idx = 0
        for xapp in XAPP_NAMES:
            for param in XAPP_CONTROLS[xapp]:
                lo, hi = PARAM_BOUNDS[param]
                delta = float(action[idx]) * ACTION_SCALE[param]
                self.params[param] = float(np.clip(self.params[param] + delta, lo, hi))
                idx += 1

        self.prev_kpis = dict(self.curr_kpis)
        self.curr_kpis = self._compute_kpis()

        self._step += 1
        return (
            self._obs(),
            self._causal_reward(),
            self._step >= self.max_steps,
            False,
            {},
        )

    def build_graph(self) -> HeteroData:
        """
        Build HeteroData matching train_hgt_kpi.py schema exactly.
        KPI node features are [norm_kpi, delta_kpi] — same 2-dim as trained on after delta change.
        This is what lets us warm-start from trained weights.
        """
        data = HeteroData()
        data["xapp"].x = torch.eye(len(XAPP_NAMES), dtype=torch.float32)
        data["param"].x = torch.tensor(
            [[self.params[p]] for p in PARAM_NAMES], dtype=torch.float32
        )

        norm = [self._norm(k, self.curr_kpis[k]) for k in KPI_NAMES]
        delta = [
            self._norm(k, self.curr_kpis[k]) - self._norm(k, self.prev_kpis[k])
            for k in KPI_NAMES
        ]
        data["kpi"].x = torch.tensor(
            [[n, d] for n, d in zip(norm, delta)], dtype=torch.float32
        )

        data[("xapp", "changes", "param")].edge_index = self._xapp_param_edges
        data[("param", "influences", "kpi")].edge_index = self._param_kpi_edges
        data[("xapp", "to", "kpi")].edge_index = self._xapp_kpi_edges
        return data


class HGTBackbone(nn.Module):

    def __init__(self, hidden: int = HGT_HIDDEN, heads: int = HGT_HEADS, layers: int = HGT_LAYERS):
        super().__init__()
        self.x_lin = nn.ModuleDict({
            "xapp":  Linear(len(XAPP_NAMES), hidden),
            "param": Linear(1,               hidden),
            "kpi":   Linear(2,               hidden),
        })
        self.convs = nn.ModuleList([
            HGTConv(hidden, hidden, metadata=HGT_METADATA, heads=heads)
            for _ in range(layers)
        ])

    def forward(self, data: HeteroData, kpi_override: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        kpi_feat = kpi_override if kpi_override is not None else data["kpi"].x

        x_dict = {
            "xapp":  self.x_lin["xapp"](data["xapp"].x),
            "param": self.x_lin["param"](data["param"].x),
            "kpi":   self.x_lin["kpi"](kpi_feat),
        }
        for conv in self.convs:
            out = conv(x_dict, data.edge_index_dict)
            x_dict = {k: (out[k] if out.get(k) is not None else x_dict[k]) for k in x_dict}
        return x_dict

class CausalGate(nn.Module):
    def __init__(self, in_dim: int = 2):
        super().__init__()
        self.gate = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))

class CausalHGTPolicyV2(nn.Module):

    def __init__(self, hidden: int = HGT_HIDDEN):
        super().__init__()
        self.backbone    = HGTBackbone(hidden=hidden)
        self.causal_gate = CausalGate(in_dim=2)

        self.actor_heads = nn.ModuleDict({
            xapp: nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, len(params))
            )
            for xapp, params in XAPP_CONTROLS.items()
        })
        self.critic = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(N_ACTIONS) - 1.0)

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
     
        kpi_gated = self.causal_gate(data["kpi"].x)

        # Pass gated features via kpi_override — backbone never sees data["kpi"].x directly
        x_dict = self.backbone(data, kpi_override=kpi_gated)

        # Actor: each xApp reads from its own observed KPI embeddings
        action_means = []
        for xapp in XAPP_NAMES:
            kpi_indices = torch.tensor(
                [KPI_NAMES.index(k) for k in XAPP_OBSERVES[xapp]], dtype=torch.long
            )
            kpi_emb = x_dict["kpi"][kpi_indices].mean(dim=0)
            action_means.append(self.actor_heads[xapp](kpi_emb))
        action_mean = torch.cat(action_means, dim=0)
        action_std  = self.log_std.exp()

        # Critic: full graph context
        xapp_pool  = x_dict["xapp"].mean(dim=0)
        param_pool = x_dict["param"].mean(dim=0)
        kpi_pool   = x_dict["kpi"].mean(dim=0)
        value = self.critic(torch.cat([xapp_pool, param_pool, kpi_pool], dim=0))

        return action_mean, action_std, value.squeeze()

def load_hgt_weights(policy: CausalHGTPolicyV2, hgt_pt_path: str) -> bool:
    """
    Load pre-trained HGT backbone weights.
    
    x_lin.kpi is EXCLUDED because the feature dim changed:
      - train_hgt_kpi.py: KPI feature = [kpi_{t-1}]          → shape [128, 1]
      - RL backbone:      KPI feature = [norm_kpi, delta_kpi] → shape [128, 2]
    
    Everything else loads fine:
      - x_lin.xapp  : [128, 4]  → [128, 4]  ✓
      - x_lin.param : [128, 1]  → [128, 1]  ✓
      - convs.*     : all dims match         ✓
    """
    if not os.path.exists(hgt_pt_path):
        print(f"[WARN] HGT weights not found at {hgt_pt_path}. Training from scratch.")
        return False

    ckpt  = torch.load(hgt_pt_path, map_location="cpu")
    saved = ckpt["model_state_dict"]

    backbone_state = {}
    skipped = []
    for k, v in saved.items():
        if k.startswith("x_lin.kpi"):          # dim mismatch — skip
            skipped.append(k)
            continue
        if k.startswith("x_lin.") or k.startswith("convs."):
            backbone_state[k] = v

    missing, unexpected = policy.backbone.load_state_dict(backbone_state, strict=False)
    print(f"[HGT] Loaded backbone weights from {hgt_pt_path}")
    print(f"      Skipped (dim mismatch): {skipped}")
    print(f"      Missing (will train):   {[m for m in missing]}")
    print(f"      Unexpected:             {unexpected}")
    return True
def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Generalised Advantage Estimation.
    Better variance reduction than plain returns, important for causal RL
    because interventional rewards can be sparse (many no-ops = 0 reward).
    """
    advantages, returns = [], []
    gae = 0.0
    next_val = 0.0
    for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_val * (1 - int(done)) - v
        gae = delta + gamma * lam * (1 - int(done)) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + v)
        next_val = v
    return advantages, returns


def train(
    hgt_pt_path: str = "SIM_OUTPUT/hgt_kpi_corrected.pt",
    n_episodes: int = 400,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 1e-4,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    max_steps: int = 300,
):
    env = ORANEnv(max_steps=max_steps)
    policy = CausalHGTPolicyV2()
    loaded = load_hgt_weights(policy, hgt_pt_path)

    # if backbone loaded, use lower LR for it to avoid catastrophic forgetting
    backbone_params = list(policy.backbone.parameters())
    head_params = [
        p
        for n, p in policy.named_parameters()
        if not any(n.startswith(x) for x in ["backbone"])
    ]
    optim = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": lr * 0.1 if loaded else lr},
            {"params": head_params, "lr": lr},
        ]
    )

    ep_rewards = []
    kpi_log: List[Dict] = []  # track per-KPI improvement over training

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        graphs, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        while True:
            graph = env.build_graph()
            with torch.no_grad():
                mu, sigma, val = policy(graph)
            dist = torch.distributions.Normal(mu, sigma)
            act = dist.sample().clamp(-1.0, 1.0)
            lp = dist.log_prob(act).sum()

            obs, rew, terminated, truncated, _ = env.step(act.numpy())

            graphs.append(graph)
            actions.append(act.detach())
            log_probs.append(lp.detach())
            values.append(float(val.detach()))
            rewards.append(rew)
            dones.append(terminated or truncated)

            if terminated or truncated:
                break

        ep_rewards.append(sum(rewards))
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)  # normalise advantages
        ret_t = torch.tensor(returns, dtype=torch.float32)

        # PPO update
        for _ in range(ppo_epochs):
            for i in range(len(graphs)):
                mu, sigma, val = policy(graphs[i])
                dist = torch.distributions.Normal(mu, sigma)
                new_lp = dist.log_prob(actions[i]).sum()
                ratio = torch.exp(new_lp - log_probs[i])
                ent = dist.entropy().sum()

                actor_loss = -torch.min(
                    ratio * adv_t[i],
                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[i],
                )
                critic_loss = F.mse_loss(val, ret_t[i])
                loss = actor_loss + 0.5 * critic_loss - 0.01 * ent

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()

        # track per-KPI state at end of episode
        kpi_log.append({k: env.curr_kpis[k] for k in KPI_NAMES})

        if ep % 20 == 0:
            recent_r = np.mean(ep_rewards[-20:])
            # show causal signal: how much did agent improve each KPI on average
            recent_kpis = kpi_log[-20:]
            kpi_summary = {k: np.mean([d[k] for d in recent_kpis]) for k in KPI_NAMES}
            print(f"\nEpisode {ep:>4} | Avg Reward: {recent_r:.4f}")
            print("  Final KPI means (last 20 eps):")
            for k, v in kpi_summary.items():
                direction = "↑" if KPI_META[k]["dir"] == +1 else "↓"
                print(f"    {k:25s}: {v:8.3f}  {direction}")

    return policy, env, ep_rewards


def evaluate(policy: CausalHGTPolicyV2, env: ORANEnv, n_episodes: int = 20):
    """
    Standard evaluation + counterfactual comparison:
    For each episode, also run a no-op baseline (action=0 always).
    The difference in reward is the causal effect of the policy's interventions.
    This approximates L3 thinking: "what would have happened without the agent?"
    """
    policy.eval()
    policy_rewards, baseline_rewards = [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            # policy rollout
            obs, _ = env.reset()
            start_params = dict(env.params)
            ep_r = 0.0
            while True:
                graph = env.build_graph()
                mu, _, _ = policy(graph)
                obs, r, term, trunc, _ = env.step(mu.numpy().clip(-1, 1))
                ep_r += r
                if term or trunc:
                    break
            policy_rewards.append(ep_r)

            env.params = start_params
            env._step = 0
            env.curr_kpis = env._compute_kpis()
            env.prev_kpis = dict(env.curr_kpis)
            base_r = 0.0
            while True:
                _, r, term, trunc, _ = env.step(np.zeros(N_ACTIONS, dtype=np.float32))
                base_r += r
                if term or trunc:
                    break
            baseline_rewards.append(base_r)

    causal_effect = np.mean(policy_rewards) - np.mean(baseline_rewards)
    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    print(
        f"Policy   reward: {np.mean(policy_rewards):.4f} ± {np.std(policy_rewards):.4f}"
    )
    print(
        f"Baseline reward: {np.mean(baseline_rewards):.4f} ± {np.std(baseline_rewards):.4f}"
    )
    print(f"Causal effect of policy (ATE): {causal_effect:+.4f}")
    print("Positive ATE = agent's interventions genuinely improved the network.")
    return policy_rewards, baseline_rewards


if __name__ == "__main__":
    policy, env, ep_rewards = train(
        hgt_pt_path="SIM_OUTPUT/hgt_kpi_corrected.pt",
        n_episodes=400,
    )
    evaluate(policy, env)
    torch.save(policy.state_dict(), "SIM_OUTPUT/causal_policy_v2.pt")
    print("\nDone. Policy saved.")
