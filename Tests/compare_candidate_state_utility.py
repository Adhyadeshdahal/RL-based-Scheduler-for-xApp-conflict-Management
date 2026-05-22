import argparse
import random

import numpy as np
import torch

from Environment.Environment_I import ORANEnvironment
from Environment.Environment_II import ORANEnvironment2
from Models.CDL import CDL
from Models.MLP import MLPInference
from Parameters import (
    CMI_THRESHOLD,
    DEVICE,
    EVAL_TAU,
    FEATURE_FC_DIMS,
    GENERATIVE_FC_DIMS,
    GRAD_CLIP,
)


SEED = 500


def state_to_tensor(state_dict):
    params, kpis = [], []
    for key in sorted(state_dict.keys()):
        if "param" in key:
            params.append(state_dict[key])
        elif "kpi" in key:
            kpis.append(state_dict[key])
    return torch.from_numpy(np.concatenate(params + kpis)).float()


def denormalize_params(norm_params, env):
    raw = np.empty_like(norm_params)
    for i, param in enumerate(env.params):
        lo, hi = param.get_threshold()
        raw[i] = norm_params[i] * (hi - lo) + lo
    return raw


def state_with_candidate_param(env, state_batch, action_batch):
    candidate_state = state_batch.clone()
    action_np = action_batch.detach().cpu().numpy()

    for row, action in enumerate(action_np):
        param_id = int(action[0])
        raw_value = env.action_to_param(
            [param_id, int(action[1]), int(action[2])]
        )[1]
        low, high = env.params[param_id].get_threshold()
        norm_value = (raw_value - low) / (high - low) if high != low else 1.0
        candidate_state[row, param_id] = float(norm_value)

    return candidate_state


class InputModeModel:
    def __init__(self, model, env, mode):
        self.model = model
        self.env = env
        self.mode = mode
        self.device = model.device

    def predictNextState(self, states, actions):
        if self.mode == "changed_state_zero_action":
            states = state_with_candidate_param(self.env, states, actions)
            actions = torch.zeros_like(actions)
        return self.model.predictNextState(states, actions)


def node_names(env):
    return [f"param{i}" for i in range(env.num_params)] + [
        kpi.name for kpi in env.kpis
    ]


def make_env(name):
    if name == "EnvironmentI":
        return ORANEnvironment()
    if name == "EnvironmentII":
        return ORANEnvironment2()
    raise ValueError(f"Unknown environment: {name}")


def make_model(kind, env):
    kwargs = dict(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        device=DEVICE,
        cmi_threshold=CMI_THRESHOLD,
        eval_tau=EVAL_TAU,
        grad_clip=GRAD_CLIP,
        generative_fc_dims=GENERATIVE_FC_DIMS,
        feature_fc_dims=FEATURE_FC_DIMS,
        lr=1e-3,
        kpi_start=env.num_params,
        node_names=node_names(env),
    )
    if kind == "CMI":
        return CDL(**kwargs)
    if kind == "MLP":
        return MLPInference(**kwargs)
    raise ValueError(f"Unknown model kind: {kind}")


def kpi_to_xapp_ids(env):
    mapping = {}
    for kpi_idx, kpi in enumerate(env.kpis):
        owner_id = None
        name = getattr(kpi, "name", "").lower()
        if name.startswith("kpi"):
            suffix = "".join(char for char in name[3:] if char.isdigit())
            if suffix:
                numbered_id = int(suffix) - 1
                if 0 <= numbered_id < len(env.xapps):
                    owner_id = numbered_id
                else:
                    grouped_id = int(suffix[0]) - 1
                    if 0 <= grouped_id < len(env.xapps):
                        owner_id = grouped_id
        if owner_id is None and kpi_idx < len(env.xapps):
            owner_id = kpi_idx
        mapping[env.num_params + kpi_idx] = [] if owner_id is None else [owner_id]
    return mapping


def detect_conflict_edges(causal_graph, env):
    num_params = env.num_params
    state_dim = causal_graph.shape[0]
    param2xapp = {}
    xapp2id = {xapp: i for i, xapp in enumerate(env.xapps)}
    param2id = {param: i for i, param in enumerate(env.params)}
    kpi2xapp = kpi_to_xapp_ids(env)

    for xapp in env.xapps:
        for param in xapp.params:
            param2xapp.setdefault(param2id[param], []).append(xapp)

    edges = []
    for kpi_node in range(num_params, state_dim):
        primary_xapp_ids = kpi2xapp.get(kpi_node, [])
        if not primary_xapp_ids:
            continue
        primary_xapp_id = primary_xapp_ids[0]

        for param_id in range(num_params):
            if causal_graph[kpi_node, param_id] != 1:
                continue
            xapps_in_conflict = param2xapp.get(param_id, [])
            if not xapps_in_conflict:
                continue
            conflict_xapp_ids = sorted(
                {xapp2id[xapp] for xapp in xapps_in_conflict} | {primary_xapp_id}
            )
            edges.append(dict(
                primary_xapp_id=primary_xapp_id,
                param_id=param_id,
                xapps_in_conflict=xapps_in_conflict,
                conflict_xapp_ids=conflict_xapp_ids,
            ))
    return edges


def normalized_threshold(env, xapp):
    xapp_idx = env.xapps.index(xapp)
    mean, std = env.kpis[xapp_idx].mean, env.kpis[xapp_idx].std
    return (xapp.threshold - mean) / std


def weighted_distance(env, xapp, utility):
    threshold = normalized_threshold(env, xapp)
    if xapp.direction == 0:
        if utility < threshold:
            return threshold - utility, 0
        return 0.0, 1

    if utility > threshold:
        return utility - threshold, 0
    return 0.0, 1


def candidate_actions(env, param_id):
    max_index = env.action_space[param_id + 2]
    actions = [
        [param_id, bin_id, index]
        for bin_id in range(env.action_space[1] + 1)
        for index in range(max_index + 1)
    ]
    return torch.tensor(actions, dtype=torch.float32)


def choose_action(model, env, state, edge, weights, tau):
    actions = candidate_actions(env, edge["param_id"]).to(model.device)
    states = state.unsqueeze(0).expand(actions.shape[0], -1).float().to(model.device)

    with torch.no_grad():
        predicted_kpis = model.predictNextState(states, actions).mean.cpu().numpy()

    costs = np.zeros(actions.shape[0])
    for row, kpis in enumerate(predicted_kpis):
        cost_vec, sat_vec = [], []
        for xapp, weight in zip(edge["xapps_in_conflict"], weights):
            utility = xapp.compute_utility(kpis)
            distance, satisfied = weighted_distance(env, xapp, utility)
            cost_vec.append(weight * distance * tau)
            sat_vec.append(satisfied)
        costs[row] = np.sum(cost_vec) - np.sum(sat_vec) ** 2

    return actions[int(np.argmin(costs))].cpu().numpy()


def compute_utility(utility_fn, raw_params, param_id, action_value):
    params = raw_params.copy()
    params[param_id] = action_value
    return float(utility_fn(params))


def load_edges(env):
    cdl = make_model("CMI", env)
    cdl.load_model(f"CMI-{env_name(env)}_model.pt")
    graph = cdl.get_binary_graph()[:, :-1].cpu().detach().numpy()
    edges = detect_conflict_edges(graph, env)
    if not edges:
        edges = detect_conflict_edges(env.true_adj_matrix, env)
    return edges


def env_name(env):
    return "EnvironmentII" if isinstance(env, ORANEnvironment2) else "EnvironmentI"


def collect_states(env, steps):
    states, raw_params = [], []
    env.reset()
    for _ in range(steps):
        state = state_to_tensor(env._get_state())
        states.append(state)
        raw_params.append(denormalize_params(state[:env.num_params].numpy(), env))
        env.step(np.zeros(env.action_dim))
    return states, raw_params


def evaluate_utility(env, model, edges, states, raw_params_by_step):
    utility_fns = env.get_utility_fns()
    utilities = []

    for state, raw_params in zip(states, raw_params_by_step):
        for edge in edges:
            num_xapps = len(edge["xapps_in_conflict"])
            weights = np.ones(num_xapps)
            weights = weights / weights.sum() * num_xapps

            action = choose_action(
                model=model,
                env=env,
                state=state,
                edge=edge,
                weights=weights.tolist(),
                tau=10,
            )
            raw_value = env.action_to_param(action.astype(int))[1]

            for xapp_id in edge["conflict_xapp_ids"]:
                if xapp_id < len(utility_fns):
                    utilities.append(compute_utility(
                        utility_fns[xapp_id],
                        raw_params,
                        edge["param_id"],
                        raw_value,
                    ))

    return float(np.mean(utilities))


def run(environment, steps):
    env = make_env(environment)
    edges = load_edges(env)
    states, raw_params_by_step = collect_states(env, steps)

    results = {}
    for model_kind in ("CMI", "MLP"):
        base_model = make_model(model_kind, env)
        base_model.load_model(f"{model_kind}-{environment}_model.pt")
        for mode in ("original_state_actual_action", "changed_state_zero_action"):
            wrapped = InputModeModel(base_model, env, mode)
            results[(model_kind, mode)] = evaluate_utility(
                env, wrapped, edges, states, raw_params_by_step
            )
    return results


def print_results(environment, steps, results):
    print(f"\nEnvironment: {environment}")
    print(f"Steps      : {steps}")
    print("\nModel  Input mode                    Mean true utility")
    print("-----  ----------------------------  -----------------")
    for model_kind in ("CMI", "MLP"):
        for mode in ("original_state_actual_action", "changed_state_zero_action"):
            print(f"{model_kind:<5}  {mode:<28}  {results[(model_kind, mode)]: .6f}")

    cmi_delta = (
        results[("CMI", "changed_state_zero_action")]
        - results[("CMI", "original_state_actual_action")]
    )
    mlp_delta = (
        results[("MLP", "changed_state_zero_action")]
        - results[("MLP", "original_state_actual_action")]
    )
    cmi_vs_mlp_changed = (
        results[("CMI", "changed_state_zero_action")]
        - results[("MLP", "changed_state_zero_action")]
    )
    print(f"\nCMI input-mode delta : {cmi_delta:+.6f}")
    print(f"MLP input-mode delta : {mlp_delta:+.6f}")
    print(f"CMI - MLP, changed   : {cmi_vs_mlp_changed:+.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="both",
                        choices=["EnvironmentI", "EnvironmentII", "both"])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    environments = (
        ["EnvironmentI", "EnvironmentII"]
        if args.environment == "both"
        else [args.environment]
    )
    for environment in environments:
        results = run(environment, args.steps)
        print_results(environment, args.steps, results)


if __name__ == "__main__":
    main()
