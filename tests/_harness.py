"""Deterministic fixtures for the golden regression lock.

Characterization only: pins current behaviour so refactors can be shown not to change
it. It does not validate against the paper.

Uses the ground-truth adjacency instead of a learned graph, and an untrained model.
A short-trained model learns an empty graph, which yields zero conflict edges and
exercises nothing; the paper checkpoints do not exist yet (Phase 0.3).

Planner hyperparameters are passed explicitly rather than read from the config, so
goldens survive config changes.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import TypedDict

import numpy as np
import torch

BASE_SEED = 12345
DEVICE = torch.device("cpu")  # CUDA kernels are not bitwise reproducible

MODEL_KWARGS = dict(
    cmi_threshold=0.2,
    eval_tau=0.99,
    grad_clip=10.0,
    generative_fc_dims=[64, 64],
    feature_fc_dims=[64, 64],
    lr=1e-3,
)


class CEMPlannerKwargs(TypedDict):
    n_candidate: int
    n_top: int
    n_iter: int


class MPPIPlannerKwargs(TypedDict):
    n_samples: int
    temperature: float
    noise_sigma: float


class MCTSPlannerKwargs(TypedDict):
    n_simulations: int
    ucb_c: float


CEM_KWARGS: CEMPlannerKwargs = dict(n_candidate=64, n_top=32, n_iter=5)
MPPI_KWARGS: MPPIPlannerKwargs = dict(n_samples=256, temperature=0.6, noise_sigma=0.1)
MCTS_KWARGS: MCTSPlannerKwargs = dict(n_simulations=8, ucb_c=1.5)
SCALING_TERM = 10  # matches cdd_oran/experiments/evaluate.py:324

ENV_NAMES = ["EnvironmentI", "EnvironmentII"]
MODEL_KINDS = ["CDL", "MLP"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(env_name: str):
    from cdd_oran.config import DEFAULT_CONFIG
    from cdd_oran.envs import get_env

    # Seed before construction: Param.__init__ draws from np.random.
    seed_all(BASE_SEED)
    env = get_env(replace(DEFAULT_CONFIG, environment=env_name))
    env.reset()
    return env


def build_model(kind: str, env):
    from cdd_oran.models.cdl import CDL
    from cdd_oran.models.mlp import MLPInference

    seed_all(BASE_SEED)
    cls = CDL if kind == "CDL" else MLPInference
    return cls(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        device=DEVICE,
        kpi_start=env.num_params,
        node_names=[f"n{i}" for i in range(env.get_state_dim())],
        **MODEL_KWARGS,
    )


def build_planners(model, env):
    from cdd_oran.planners.cem import ModelBasedCEM
    from cdd_oran.planners.mcts import ModelBasedMCTS
    from cdd_oran.planners.mppi import ModelBasedMPPI
    from cdd_oran.planners.qacm import QACM

    return [
        QACM(model=model, env=env),
        ModelBasedCEM(model=model, env=env, **CEM_KWARGS),
        ModelBasedMPPI(model=model, env=env, **MPPI_KWARGS),
        ModelBasedMCTS(model=model, env=env, **MCTS_KWARGS),
    ]


def _cost_probe(qacm, env) -> list:
    """Probe the cost function directly.

    Action-level assertions are not sensitive enough on their own: planners argmin over
    a discrete grid, so small cost drift leaves every action unchanged (verified --
    scaling the normalised threshold by 1.05 changes no action).
    """
    probe = []
    for xapp_idx, xapp in enumerate(env.xapps):
        for u in (-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0):
            d, s = qacm.obtain_weighted_distance(xapp, u)
            probe.append([xapp_idx, u, round(float(d), 10), int(s)])
    return probe


def _model_probe(model, env, state_t) -> list:
    """Continuous-valued probe of the world model's forward pass."""
    s = state_t.unsqueeze(0).to(DEVICE)
    a = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32, device=DEVICE)
    seed_all(BASE_SEED)
    with torch.no_grad():
        dist = model.predict_next_state(s, a)
    return [
        [round(float(v), 8) for v in dist.mean.flatten().tolist()],
        [round(float(v), 8) for v in dist.scale.flatten().tolist()],
    ]


def record(env_name: str, model_kind: str) -> dict:
    """Produce the full deterministic record for one (env, model) pair."""
    from cdd_oran.conflicts import (
        compute_utility,
        denormalize_params,
        detect_conflict_edges,
        state_to_tensor,
    )

    env = build_env(env_name)
    model = build_model(model_kind, env)
    planners = build_planners(model, env)

    state_t = state_to_tensor(env.get_state()).to(DEVICE)
    raw_params = denormalize_params(state_t[: env.num_params].cpu().numpy(), env)
    utility_fns = env.get_utility_fns()

    edges = detect_conflict_edges(env.true_adj_matrix, env)

    out = {
        "env": env_name,
        "model_kind": model_kind,
        "state_dim": int(env.get_state_dim()),
        "action_dim": int(env.get_action_dim()),
        "num_params": int(env.num_params),
        "true_adj_edges": int(env.true_adj_matrix.sum()),
        "num_conflict_edges": len(edges),
        "state": [round(float(v), 10) for v in state_t.tolist()],
        "raw_params": [round(float(v), 10) for v in raw_params.tolist()],
        "cost_probe": _cost_probe(planners[0], env),
        "model_probe": _model_probe(model, env, state_t),
        "results": [],
    }

    for edge_idx, edge in enumerate(edges):
        param_id = edge["param_id"]
        xapps = edge["xapps_in_conflict"]
        weights = (np.ones(len(xapps)) / len(xapps) * len(xapps)).tolist()

        for planner in planners:
            # Re-seed per call so results don't depend on planner ordering.
            seed_all(BASE_SEED)
            action = planner.act(
                current_state=state_t.clone().detach(),
                conflict_param_index=param_id,
                xapps_under_conflict=xapps,
                weights_per_xapps=weights,
                scaling_term=SCALING_TERM,
            )
            raw_val = env.action_to_param(action)[1]
            utils = [
                round(float(compute_utility(utility_fns[x], raw_params, param_id, raw_val)), 8)
                for x in edge["conflict_xapp_ids"]
                if x < len(utility_fns)
            ]
            out["results"].append(
                {
                    "edge": edge_idx,
                    "param_id": int(param_id),
                    "primary_xapp_id": int(edge["primary_xapp_id"]),
                    "planner": planner.name,
                    "action": [int(a) for a in action],
                    "raw_val": round(float(raw_val), 8),
                    "utilities": utils,
                }
            )

    return out
