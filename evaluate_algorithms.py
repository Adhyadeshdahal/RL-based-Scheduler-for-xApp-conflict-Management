"""
evaluate.py  –  xApp Utility Curves with Algorithm Parameter Selections
------------------------------------------------------------------------
Pipeline
  1. For N environment steps:
     a. Get current state from env
     b. Use learned CDL module to detect conflict edges (kpi_node → param_id)
     c. For each conflict edge, run every algorithm → get action → compute utility
     d. Log mean utility per algorithm per step to TensorBoard (all on ONE graph)
     e. Step the environment forward
  2. TensorBoard logs:
     - Mean utility per algorithm per environment step (single graph, different colours)
     - Conflict count per step
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import rcParams
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from Algorithms import get_algorithms
from Models.CDL import CDL
from Parameters import *
from Environment import get_env
from Models import get_model

SEED      = 500
NUM_STEPS = 47
IS_TRAIN  = False
IS_TRAIN = True   # Set to False to run evaluation without training loop (if needed)

XAPP_COLORS = [
    "#E91E8C",
    "#00BCD4",
    "#4CAF50",
    "#FF9800",
    "#3F51B5",
]

ALGO_STYLES = [
    dict(color="#D32F2F", marker="o",  linestyle="-",  label="QACM"),
    dict(color="#212121", marker="D",  linestyle="-",  label="ModelBasedMCTS"),
    dict(color="#F44336", marker="^",  linestyle="--", label="ModelBasedMPPI"),
    dict(color="#00897B", marker="s",  linestyle="-.", label="ModelBasedCEM"),
]
_EXTRA_COLORS  = ["#7B1FA2", "#1565C0", "#558B2F", "#E65100"]
_EXTRA_MARKERS = ["P", "X", "v", "<"]

THRESHOLD_ALPHA    = 0.65
GRID_COLOR         = "#E0E0E0"
BG_COLOR           = "white"
JITTER_THRESHOLD_FRAC = 0.01
JITTER_STEP_FRAC      = 0.015

rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.facecolor":     BG_COLOR,
    "figure.facecolor":   BG_COLOR,
    "axes.edgecolor":     "#BDBDBD",
    "axes.grid":          True,
    "grid.color":         GRID_COLOR,
    "grid.linewidth":     0.6,
    "grid.alpha":         1.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.color":        "#424242",
    "ytick.color":        "#424242",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.labelsize":     10,
    "axes.labelcolor":    "#212121",
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.titlecolor":    "#212121",
})


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
        raw[i]  = norm_params[i] * (hi - lo) + lo
    return raw


def algo_style(idx, algo_names):
    if idx < len(ALGO_STYLES):
        return ALGO_STYLES[idx]
    extra = idx - len(ALGO_STYLES)
    return dict(
        color     = _EXTRA_COLORS[extra % len(_EXTRA_COLORS)],
        marker    = _EXTRA_MARKERS[extra % len(_EXTRA_MARKERS)],
        linestyle = "-",
        label     = algo_names[idx] if idx < len(algo_names) else f"Algo {idx}",
    )


def jitter_values(raw_vals, sweep_range,
                  threshold_frac=JITTER_THRESHOLD_FRAC,
                  step_frac=JITTER_STEP_FRAC):
    min_gap = threshold_frac * sweep_range
    step    = step_frac      * sweep_range
    display = list(raw_vals)
    indexed = sorted(enumerate(raw_vals), key=lambda t: t[1])
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and (indexed[j][1] - indexed[i][1]) < min_gap:
            j += 1
        group = indexed[i:j]
        if len(group) > 1:
            centre  = np.mean([v for _, v in group])
            offsets = np.arange(len(group)) * step
            offsets -= offsets.mean()
            for k, (orig_idx, _) in enumerate(group):
                display[orig_idx] = centre + offsets[k]
        i = j
    return display


def detect_conflict_edges(causal_graph, env):
    num_params  = env.num_params
    state_dim   = causal_graph.shape[0]
    param2xapp  = {}
    xapp2id     = {xa: i for i, xa in enumerate(env.xapps)}
    param2id    = {p: i for i, p in enumerate(env.params)}

    for xapp in env.xapps:
        for p in xapp.params:
            pid = param2id[p]
            param2xapp.setdefault(pid, []).append(xapp)

    edges = []
    for kpi_node in range(num_params, state_dim):
        primary_xapp_id = kpi_node - num_params
        if primary_xapp_id >= len(env.xapps):
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
            edges.append(dict(
                primary_xapp_id   = primary_xapp_id,
                param_id          = param_id,
                xapps_in_conflict = xapps_in_conflict,
                conflict_xapp_ids = conflict_xapp_ids,
            ))
    return edges


def compute_utility(utility_fn, raw_params, param_id, action_val):
    p = raw_params.copy()
    p[param_id] = action_val
    return float(utility_fn(p))


def draw_panel(ax, sweep, curves, algo_actions, algo_names,
               param_id, primary_xapp_id, thresh,
               KPI_THRESHOLDS, MEAN_STD_KPIS):
    sweep_range = thresh[1] - thresh[0]

    for xapp_id, values in curves:
        color = XAPP_COLORS[xapp_id % len(XAPP_COLORS)]
        lw    = 2.2 if xapp_id == primary_xapp_id else 1.5
        ax.plot(sweep, values, color=color, linewidth=lw, zorder=3)

    for xapp_id, _ in curves:
        if xapp_id >= len(KPI_THRESHOLDS):
            continue
        raw_thresh = KPI_THRESHOLDS[xapp_id]
        mean, std  = MEAN_STD_KPIS[xapp_id]
        q_i        = (raw_thresh - mean) / std
        color      = XAPP_COLORS[xapp_id % len(XAPP_COLORS)]
        ax.axhline(q_i, color=color, linewidth=1.2, linestyle="--",
                   alpha=THRESHOLD_ALPHA, zorder=2)

    primary_vals = next(v for xid, v in curves if xid == primary_xapp_id)
    peak_idx     = int(np.argmax(primary_vals))
    eta_val      = float(primary_vals[peak_idx])
    ax.axhline(eta_val, color="#9E9E9E", linewidth=1.0,
               linestyle="--", alpha=0.8, zorder=2)
    ax.scatter([sweep[peak_idx]], [eta_val],
               color="#F44336", s=55, zorder=6, linewidths=0)

    clamped = [float(np.clip(v, thresh[0], thresh[1])) for v in algo_actions]
    display = jitter_values(clamped, sweep_range)

    for idx, (disp_val, raw_val) in enumerate(zip(display, algo_actions)):
        style = algo_style(idx, algo_names)
        y_at  = float(np.interp(disp_val, sweep, primary_vals))
        ax.axvline(disp_val, color=style["color"], linewidth=1.6,
                   linestyle=style["linestyle"], alpha=0.92, zorder=5)
        if style["marker"]:
            ax.scatter([disp_val], [y_at],
                       color=style["color"], marker=style["marker"],
                       s=55, zorder=7, linewidths=0)
        ax.text(disp_val, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else
                ax.dataLim.y0,
                f" {raw_val:.1f}", color=style["color"],
                fontsize=7, va="bottom", ha="center", zorder=8,
                rotation=90, clip_on=True)

    ax.set_xlim(thresh[0], thresh[1])
    ax.set_xlabel(f"Values for $p_{{{param_id}}}$", labelpad=4)
    ax.set_ylabel(f"Values for $U(p_{{{param_id}}})$", labelpad=4)
    others     = [xid for xid, _ in curves if xid != primary_xapp_id]
    others_str = ", ".join(f"$x_{{{i}}}$" for i in others)
    title      = (f"$x_{{{primary_xapp_id}}}$ conflict with {others_str} "
                  f"over $p_{{{param_id}}}$") if others else \
                 f"$x_{{{primary_xapp_id}}}$ over $p_{{{param_id}}}$"
    ax.set_title(title, pad=7)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env       = get_env()
    state_dim = env.get_state_dim()
    act_dim   = env.get_action_dim()
    model     = get_model(env)

    cdl = CDL(
        state_dim          = state_dim,
        action_dim         = act_dim,
        device             = DEVICE,
        cmi_threshold      = CMI_THRESHOLD,
        eval_tau           = EVAL_TAU,
        grad_clip          = GRAD_CLIP,
        generative_fc_dims = GENERATIVE_FC_DIMS,
        feature_fc_dims    = FEATURE_FC_DIMS,
        lr                 = 1e-3,
        kpi_start          = env.num_params,
    )
    try:
        cdl.load_model(CDL_LOAD_NAME)
    except FileNotFoundError:
        print("[ERROR] CDL model not found:", CDL_LOAD_NAME)
        return -1

    try:
        model.load_model(MODEL_LOAD_NAME)
    except FileNotFoundError:
        print("[ERROR] Inference model not found:", MODEL_LOAD_NAME)
        return -1

    algorithms  = get_algorithms(model, env)
    algo_names  = [a.name for a in algorithms]
    utility_fns = env.get_utility_fns()

    KPI_THRESHOLDS, MEAN_STD_KPIS = env.get_thresholds_stds()

    tag_suffix = "CMI" if USE_CMI else "MLP"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir    = f"runs/{ENVIRONMENT}/evaluate_{tag_suffix}_{timestamp}"
    writer     = SummaryWriter(log_dir=run_dir)

    print(f"TensorBoard logging to:  {run_dir}")
    print(f"Running evaluation for {NUM_STEPS} environment steps ...\n")

    env.reset()

    for global_step in range(NUM_STEPS):

        state_dict = env._get_state()
        state_t    = state_to_tensor(state_dict).to(DEVICE)
        raw_params = denormalize_params(
            state_t[:env.num_params].cpu().numpy(), env
        )

        causal_graph = cdl.get_binary_graph()[:, :-1].cpu().detach().numpy()
        edges        = detect_conflict_edges(causal_graph, env)

        num_conflicts = len(edges)
        writer.add_scalar("conflicts/count", num_conflicts, global_step)

        if num_conflicts == 0:
            print(f"[step {global_step:4d}]  No conflicts detected.")
        else:
            print(f"[step {global_step:4d}]  {num_conflicts} conflict(s) detected.")

        step_utilities = {algo.name: [] for algo in algorithms}

        for edge in edges:
            param_id          = edge["param_id"]
            primary_xapp_id   = edge["primary_xapp_id"]
            xapps_in_conflict = edge["xapps_in_conflict"]
            conflict_xapp_ids = edge["conflict_xapp_ids"]

            param  = env.params[param_id]
            thresh = param.get_threshold()

            num_xapps = len(xapps_in_conflict)
            weights   = np.ones(num_xapps)
            for idx, xapp in enumerate(xapps_in_conflict):
                if xapp == env.xapps[primary_xapp_id]:
                    weights[idx] *= 1.0
            weights = weights / weights.sum() * num_xapps

            for algo_idx, algo in enumerate(algorithms):
                state_for_algo = state_t.clone().detach()

                action  = algo.act(
                    current_state        = state_for_algo,
                    conflict_param_index = param_id,
                    xapps_under_conflict = xapps_in_conflict,
                    weights_per_xapps    = weights.tolist(),
                    scaling_term         = 10,
                )
                raw_val = env.action_to_param(action)[1]

                for xid in conflict_xapp_ids:
                    if xid >= len(utility_fns):
                        continue
                    u = compute_utility(utility_fns[xid], raw_params,
                                        param_id, raw_val)
                    step_utilities[algo.name].append(u)

                print(f"    [{algo.name:22s}]  "
                      f"x{primary_xapp_id}→p{param_id}  "
                      f"action={raw_val:.4f}  "
                      f"utility(primary)="
                      f"{compute_utility(utility_fns[primary_xapp_id], raw_params, param_id, raw_val):.4f}"
                      if primary_xapp_id < len(utility_fns) else
                      f"    [{algo.name:22s}]  x{primary_xapp_id}→p{param_id}  action={raw_val:.4f}")

        scalars_for_step = {}
        for algo in algorithms:
            vals = step_utilities[algo.name]
            if vals:
                mean_u = float(np.mean(vals))
                scalars_for_step[algo.name] = mean_u
                print(f"  [mean utility] {algo.name:22s} → {mean_u:.4f}")

        if scalars_for_step:
            writer.add_scalars("mean_utility/all_algorithms", scalars_for_step, global_step)

        neutral_action = np.zeros(act_dim)
        env.step(neutral_action)

        writer.flush()

    writer.close()
    print(f"\nDone. Run:  tensorboard --logdir {run_dir}")
    return 0


if __name__ == "__main__":
    main()