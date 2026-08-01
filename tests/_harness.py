"""Deterministic fixtures for the golden regression lock.

Characterization only: pins current behaviour so refactors can be shown not to change
it. It does not validate against the paper.

Uses the ground-truth adjacency instead of a learned graph, and an untrained model.
A short-trained model learns an empty graph, which yields zero conflict edges and
exercises nothing; the paper checkpoints do not exist yet (Phase 0.3).

Planner hyperparameters are passed explicitly rather than taken from their
`Parameters`-sourced defaults, so goldens survive Phase 1's config rewrite.
"""

from __future__ import annotations

import random

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
PLANNER_KWARGS = {
    "ModelBasedCEM": dict(n_candidate=64, n_top=32, n_iter=5),
    "ModelBasedMPPI": dict(n_samples=256, temperature=0.6, noise_sigma=0.1),
    "ModelBasedMCTS": dict(n_simulations=8, ucb_c=1.5),
}
SCALING_TERM = 10  # matches evaluate_algorithms.py:324

ENV_NAMES = ["EnvironmentI", "EnvironmentII"]
MODEL_KINDS = ["CDL", "MLP"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(env_name: str):
    from Environment.Environment_I import ORANEnvironment
    from Environment.Environment_II import ORANEnvironment2

    # Seed before construction: Param.__init__ draws from np.random.
    seed_all(BASE_SEED)
    env = ORANEnvironment() if env_name == "EnvironmentI" else ORANEnvironment2()
    env.reset()
    return env


def build_model(kind: str, env):
    from Models.CDL import CDL
    from Models.MLP import MLPInference

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
    from Algorithms.model_based_cem import ModelBasedCEM
    from Algorithms.model_based_mcts import ModelBasedMCTS
    from Algorithms.model_based_mppi import ModelBasedMPPI
    from Algorithms.QACM import QACM

    return [
        QACM(model=model, env=env),
        ModelBasedCEM(model=model, env=env, **PLANNER_KWARGS["ModelBasedCEM"]),
        ModelBasedMPPI(model=model, env=env, **PLANNER_KWARGS["ModelBasedMPPI"]),
        ModelBasedMCTS(model=model, env=env, **PLANNER_KWARGS["ModelBasedMCTS"]),
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
        dist = model.predictNextState(s, a)
    return [
        [round(float(v), 8) for v in dist.mean.flatten().tolist()],
        [round(float(v), 8) for v in dist.scale.flatten().tolist()],
    ]


def record(env_name: str, model_kind: str) -> dict:
    """Produce the full deterministic record for one (env, model) pair."""
    from evaluate_algorithms import (
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
