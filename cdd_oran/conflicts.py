import numpy as np
import torch


def state_to_tensor(state_dict):
    kpi_vals, param_vals = [], []
    for key in sorted(state_dict.keys()):
        if "param" in key:
            param_vals.append(state_dict[key])
        elif "kpi" in key:
            kpi_vals.append(state_dict[key])
    return torch.from_numpy(np.concatenate(param_vals + kpi_vals)).float()


def denormalize_params(norm_params, env):
    raw = np.empty_like(norm_params)
    for i, p in enumerate(env.params):
        lo, hi = p.get_threshold()
        raw[i] = norm_params[i] * (hi - lo) + lo
    return raw


def detect_conflict_edges(causal_graph, env):
    num_params = env.num_params
    state_dim = causal_graph.shape[0]
    param2xapp = {}
    xapp2id = {xa: i for i, xa in enumerate(env.xapps)}
    param2id = {p: i for i, p in enumerate(env.params)}

    for xapp in env.xapps:
        for p in xapp.params:
            pid = param2id[p]
            param2xapp.setdefault(pid, []).append(xapp)

    kpi2xapp_mapping = env.get_kpi_to_xapp_mapping()

    edges = []
    for kpi_node in range(num_params, state_dim):
        primary_xapp_id = kpi2xapp_mapping.get(kpi_node)
        if primary_xapp_id is None or primary_xapp_id >= len(env.xapps):
            continue
        for param_id in range(num_params):
            if causal_graph[kpi_node, param_id] != 1:
                continue
            xapps_in_conflict = param2xapp.get(param_id, [])
            if not xapps_in_conflict:
                continue
            conflict_xapp_ids = sorted(
                {xapp2id[xa] for xa in xapps_in_conflict} | {primary_xapp_id}
            )
            edges.append(
                dict(
                    primary_xapp_id=primary_xapp_id,
                    param_id=param_id,
                    xapps_in_conflict=xapps_in_conflict,
                    conflict_xapp_ids=conflict_xapp_ids,
                )
            )
    return edges


def compute_utility(utility_fn, raw_params, param_id, action_val):
    p = raw_params.copy()
    p[param_id] = action_val
    return float(utility_fn(p))
