"""
evaluate_multi_step.py  –  N-Step Algorithm Utility Evaluation & Ranking
-------------------------------------------------------------------------
Logic (corrected)
  For each step t = 1 … N:
    1. Capture the current environment state (same tensor for every algo)
    2. For every algorithm:
         For every conflict edge detected by CDL:
             Ask the algorithm for its action on that conflict
             Compute aggregate utility of that action (sum over all xApp
             utility functions evaluated at the chosen parameter value)
         Average that utility over ALL conflict edges → one scalar per algo
    3. Advance env once (reference = first algo's action on first conflict)
       so all algorithms share the identical state trajectory.

  After N steps → utility matrix  [N × num_algos]  → rank + visualise.

Fairness guarantees
  • env.reset()  called exactly once.
  • Each algo.act() receives its own  clone().detach()  of the state tensor.
  • env.step()   called once per outer step with a single reference action.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib import rcParams

from Algorithms import get_algorithms
from Models.CDL import CDL
from Parameters import *
from Environment import get_env
from Models import get_model


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

N_STEPS     = 5      # number of evaluation steps
UTILITY_AGG = "sum"    # "sum" | "mean"  how to aggregate xApp utilities per conflict

# ── Palette ───────────────────────────────────────────────────────────────────

ALGO_PALETTE   = ["#E91E63", "#00BCD4", "#FF9800", "#4CAF50",
                  "#7C3AED", "#F44336", "#0288D1", "#FFC107"]
ALGO_MARKERS   = ["o", "D", "^", "s", "P", "X", "v", "<"]
ALGO_LINESTYLE = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

BG_COLOR     = "#0F1117"
GRID_COLOR   = "#2A2D3A"
TEXT_COLOR   = "#E8EAF0"
MEDAL_COLORS = ["#FFD700", "#C0C0C0", "#CD7F32"]

rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   "#181B25",
    "figure.facecolor": BG_COLOR,
    "axes.edgecolor":   GRID_COLOR,
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "grid.linewidth":   0.5,
    "grid.alpha":       1.0,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.labelsize":   11,
    "axes.labelcolor":  TEXT_COLOR,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.titlecolor":  TEXT_COLOR,
    "text.color":       TEXT_COLOR,
})


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

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


def compute_conflict_utility(raw_params, param_id, chosen_raw_val,
                              conflict_xapp_ids, utility_fns, mode="sum"):
    """
    Set param[param_id] = chosen_raw_val, then evaluate every conflicting
    xApp's utility function and aggregate.
    """
    p = raw_params.copy()
    p[param_id] = chosen_raw_val
    vals = [float(utility_fns[xid](p))
            for xid in conflict_xapp_ids if xid < len(utility_fns)]
    if not vals:
        return 0.0
    return float(np.sum(vals) if mode == "sum" else np.mean(vals))


def detect_conflict_edges(causal_graph, env):
    num_params = env.num_params
    state_dim  = causal_graph.shape[0]
    param2xapp = {}
    xapp2id    = {xa: i for i, xa in enumerate(env.xapps)}
    param2id   = {p:  i for i, p  in enumerate(env.params)}

    for xapp in env.xapps:
        for p in xapp.params:
            param2xapp.setdefault(param2id[p], []).append(xapp)

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
            # Primary xApp gets 2× weight
            num_x   = len(xapps_in_conflict)
            weights = np.ones(num_x)
            for idx, xapp in enumerate(xapps_in_conflict):
                if xapp == env.xapps[primary_xapp_id]:
                    weights[idx] *= 2.0
            weights = (weights / weights.sum() * num_x).tolist()

            edges.append(dict(
                primary_xapp_id   = primary_xapp_id,
                param_id          = param_id,
                xapps_in_conflict = xapps_in_conflict,
                conflict_xapp_ids = conflict_xapp_ids,
                weights           = weights,
            ))
    return edges


def algo_style(idx):
    return dict(
        color     = ALGO_PALETTE[idx % len(ALGO_PALETTE)],
        marker    = ALGO_MARKERS[idx % len(ALGO_MARKERS)],
        linestyle = ALGO_LINESTYLE[idx % len(ALGO_LINESTYLE)],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_multi_step_evaluation(env, algorithms, cdl, utility_fns, n_steps=N_STEPS):
    """
    Returns
    -------
    utility_matrix : np.ndarray  shape (n_steps, n_algos)
        Each entry = mean utility across ALL conflict edges at that step.
    algo_names     : list[str]
    n_conflicts    : int
    """
    algo_names = [a.name for a in algorithms]
    n_algos    = len(algorithms)

    causal_graph = cdl.get_binary_graph()[:, :-1].cpu().detach().numpy()
    edges        = detect_conflict_edges(causal_graph, env)

    if not edges:
        raise RuntimeError("No conflict edges found in causal graph.")

    n_conflicts = len(edges)
    print(f"\n{'═'*62}")
    print(f"  Multi-step evaluation")
    print(f"  Steps      : {n_steps}")
    print(f"  Algorithms : {n_algos}  →  {algo_names}")
    print(f"  Conflicts  : {n_conflicts}  "
          f"(utility averaged across ALL conflicts per step)")
    print(f"{'═'*62}")

    env.reset()
    utility_matrix = np.zeros((n_steps, n_algos))

    for step in range(n_steps):

        # ── Freeze current state once for this step ───────────────────────
        current_state_t = state_to_tensor(env._get_state()).to(DEVICE).clone()
        raw_params_base = denormalize_params(
            current_state_t[:env.num_params].cpu().numpy(), env
        )

        reference_action = None   # for env.step() at end of step

        # ── For each algorithm ────────────────────────────────────────────
        for algo_idx, algo in enumerate(algorithms):
            conflict_utilities = []   # one value per conflict edge

            # ── For each conflict edge ────────────────────────────────────
            for edge_idx, edge in enumerate(edges):
                # Every algo × every conflict gets a fresh clone
                state_clone = current_state_t.clone().detach()

                action = algo.act(
                    current_state        = state_clone,
                    conflict_param_index = edge["param_id"],
                    xapps_under_conflict = edge["xapps_in_conflict"],
                    weights_per_xapps    = edge["weights"],
                    scaling_term         = 10,
                )

                chosen_raw_val = env.action_to_param(action)[1]

                u = compute_conflict_utility(
                    raw_params        = raw_params_base,
                    param_id          = edge["param_id"],
                    chosen_raw_val    = chosen_raw_val,
                    conflict_xapp_ids = edge["conflict_xapp_ids"],
                    utility_fns       = utility_fns,
                    mode              = UTILITY_AGG,
                )
                conflict_utilities.append(u)

                # Reference action = first algo, first conflict
                if algo_idx == 0 and edge_idx == 0:
                    reference_action = action

            # ── Average across all conflicts → single scalar this step ────
            utility_matrix[step, algo_idx] = float(np.mean(conflict_utilities))

        # ── Advance environment once (shared trajectory) ──────────────────
        if reference_action is not None:
            try:
                env.step(reference_action)
            except Exception:
                env.reset()

        # Progress log every 50 steps
        if (step + 1) % 50 == 0 or step == 0:
            vals = "  ".join(
                f"{algo_names[i]}: {utility_matrix[:step+1, i].mean():+.3f}"
                for i in range(n_algos)
            )
            print(f"  step {step+1:>4}/{n_steps}  |  running mean  {vals}")

    print(f"{'═'*62}\n")
    return utility_matrix, algo_names, n_conflicts


# ══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════════════════════════

def smooth(arr, window=15):
    if window <= 1:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")


def draw_step_panel(ax, utility_matrix, algo_names, n_steps, n_conflicts):
    sw    = max(1, n_steps // 25)
    steps = np.arange(1, n_steps + 1)

    for i, name in enumerate(algo_names):
        s   = algo_style(i)
        raw = utility_matrix[:, i]
        smt = smooth(raw, sw)

        # Faint raw signal
        ax.plot(steps, raw, color=s["color"], alpha=0.12,
                linewidth=0.7, zorder=2)
        # Solid smoothed signal
        ax.plot(steps, smt, color=s["color"], alpha=0.92,
                linewidth=1.8, linestyle=s["linestyle"],
                label=name, zorder=4)

    ax.set_xlabel("Step", labelpad=5)
    ax.set_ylabel(
        f"Utility (mean across {n_conflicts} conflict(s),  agg={UTILITY_AGG})",
        labelpad=5
    )
    ax.set_title(
        f"Per-Step Utility — averaged over ALL {n_conflicts} conflict edge(s)",
        pad=9
    )
    ax.set_xlim(1, n_steps)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    leg = ax.legend(fontsize=8.5, loc="upper right",
                    frameon=True, framealpha=0.85,
                    facecolor="#1E2130", edgecolor=GRID_COLOR,
                    handlelength=2.0, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(TEXT_COLOR)


def draw_ranking_panel(ax, utility_matrix, algo_names, n_steps, n_conflicts):
    means = utility_matrix.mean(axis=0)
    stds  = utility_matrix.std(axis=0)
    order = np.argsort(means)[::-1]

    r_names  = [algo_names[i] for i in order]
    r_means  = means[order]
    r_stds   = stds[order]
    r_colors = [ALGO_PALETTE[i % len(ALGO_PALETTE)] for i in order]
    y_pos    = np.arange(len(order))
    span     = max(r_means.max() - r_means.min(), 1e-9)

    ax.barh(y_pos, r_means,
            xerr=r_stds,
            color=r_colors, edgecolor="none",
            height=0.55, alpha=0.88,
            error_kw=dict(ecolor=TEXT_COLOR, elinewidth=1.1,
                          capsize=3, alpha=0.7),
            zorder=3)

    for i, (m, sd) in enumerate(zip(r_means, r_stds)):
        ax.text(m + sd + span * 0.01, i,
                f"{m:.3f} ± {sd:.3f}",
                va="center", ha="left", fontsize=8,
                color=TEXT_COLOR, zorder=5)

    for i, rank in enumerate(range(1, len(order) + 1)):
        bc = MEDAL_COLORS[rank - 1] if rank <= 3 else "#3A3D4E"
        tc = "#111" if rank <= 2 else TEXT_COLOR
        ax.text(0.01, i, f" #{rank} ",
                transform=ax.get_yaxis_transform(),
                va="center", ha="left",
                fontsize=8.5, fontweight="bold", color=tc,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=bc, edgecolor="none", alpha=0.95),
                zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(r_names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Mean Utility (avg over all {n_conflicts} conflict(s))  ±  1 std  "
        f"|  {n_steps} steps",
        labelpad=5
    )
    ax.set_title("Algorithm Ranking", pad=9)
    ax.axvline(r_means[0], color=MEDAL_COLORS[0],
               linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)
    ax.set_xlim(
        r_means.min() - span * 0.08,
        r_means.max() + r_stds[0] + span * 0.40,
    )


def build_figure(utility_matrix, algo_names, n_steps, n_conflicts):
    n_algos    = len(algo_names)
    bar_height = max(4.0, n_algos * 0.8)

    fig = plt.figure(figsize=(14, 6.5 + bar_height), facecolor=BG_COLOR)
    fig.suptitle(
        f"Multi-Step Algorithm Utility  ·  {n_steps} Steps  ·  "
        f"{n_conflicts} Conflict Edge(s)  ·  utility averaged per step",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.998,
    )

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[5, bar_height],
        hspace=0.52,
        top=0.97, bottom=0.06,
        left=0.09, right=0.96,
    )

    draw_step_panel(fig.add_subplot(gs[0]),
                    utility_matrix, algo_names, n_steps, n_conflicts)
    draw_ranking_panel(fig.add_subplot(gs[1]),
                       utility_matrix, algo_names, n_steps, n_conflicts)

    fig.text(
        0.5, 0.505,
        f"Per-step value = mean( utility(conflict_1), …, utility(conflict_{n_conflicts}) )  "
        f"•  xApp utility agg = {UTILITY_AGG}  •  "
        f"Shared env trajectory  •  smoothing window = {max(1, n_steps//25)} steps",
        ha="center", va="center", fontsize=7.5, color="#6B7280",
    )

    fname = f"multi_step_utility_N{n_steps}_{'CMI' if USE_CMI else 'MLP'}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved  →  {fname}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env   = get_env()
    model = get_model(env)

    cdl = CDL(
        state_dim          = env.get_state_dim(),
        action_dim         = env.get_action_dim(),
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
        print("[ERROR] CDL model not found:", CDL_LOAD_NAME); return -1

    try:
        model.load_model(MODEL_LOAD_NAME)
    except FileNotFoundError:
        print("[ERROR] Inference model not found:", MODEL_LOAD_NAME); return -1

    algorithms  = get_algorithms(model, env)
    utility_fns = env.get_utility_fns()

    utility_matrix, algo_names, n_conflicts = run_multi_step_evaluation(
        env, algorithms, cdl, utility_fns, n_steps=N_STEPS
    )

    # ── Print summary table ───────────────────────────────────────────────────
    means = utility_matrix.mean(axis=0)
    stds  = utility_matrix.std(axis=0)
    order = np.argsort(means)[::-1]

    print("=" * 60)
    print(f"  FINAL RANKING  ({N_STEPS} steps, {n_conflicts} conflict(s))")
    print("=" * 60)
    print(f"  {'Rank':<5}  {'Algorithm':<22}  {'Mean':>9}  {'Std':>9}")
    print(f"  {'─'*4}  {'─'*22}  {'─'*9}  {'─'*9}")
    for rank, idx in enumerate(order, 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"   #{rank}"
        print(f"  {medal}   {algo_names[idx]:<22}  "
              f"{means[idx]:>+9.4f}  {stds[idx]:>9.4f}")
    print("=" * 60)

    build_figure(utility_matrix, algo_names, N_STEPS, n_conflicts)
    return 0


if __name__ == "__main__":
    main()