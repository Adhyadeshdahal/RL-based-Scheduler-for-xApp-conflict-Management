"""
evaluate.py  –  xApp Utility Curves with Algorithm Parameter Selections
------------------------------------------------------------------------
Pipeline
  1. Load CDL causal graph  →  detect conflict edges (kpi_node → param_id)
  2. For each conflict edge, collect all xApps that share that parameter
  3. Ask every algorithm for its chosen action on that (param, xApps) pair
  4. Plot utility curves + algorithm selections, one panel per conflict edge

Guaranteed fairness
  • env.reset() is called exactly once.  The resulting state tensor is
    cloned into `frozen_state_t` and never mutated again.
  • Each algo.act() receives its own clone().detach() of that tensor, so
    no algorithm can corrupt the shared buffer or see side-effects from a
    previous algorithm's forward pass.
  • env.step() is never called during evaluation — the environment state
    is the same for every algorithm across every conflict edge.

Fix for hidden algorithm lines
  • Clamp each algorithm's selected value to [thresh_lo, thresh_hi] before
    plotting so axvline never silently clips outside xlim.
  • Apply a tiny x-offset (jitter) when two or more algorithms land within
    1 % of the sweep range of each other, so their lines don't overlap.
  • Each algorithm gets a distinct colour + marker combination that is also
    used in the global legend, making every selection uniquely identifiable.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import rcParams

from Algorithms import get_algorithms
from Models.CDL import CDL
from Parameters import *
from Environment import get_env
from Models import get_model


# ── Palette ───────────────────────────────────────────────────────────────────

XAPP_COLORS = [
    "#E91E8C",   # xApp 0  magenta
    "#00BCD4",   # xApp 1  cyan
    "#4CAF50",   # xApp 2  green
    "#FF9800",   # xApp 3  orange
    "#3F51B5",   # xApp 4  indigo
]

# One entry per algorithm – colours must be visually distinct from XAPP_COLORS
# so vertical lines never blend into utility curves.
ALGO_STYLES = [
    dict(color="#D32F2F", marker="o",  linestyle="-",  label="QACM"),
    dict(color="#212121", marker="D",  linestyle="-",  label="ModelBasedMCTS"),
    dict(color="#F44336", marker="^",  linestyle="--", label="ModelBasedMPPI"),
    dict(color="#00897B", marker="s",  linestyle="-.", label="ModelBasedCEM"),
]
# Fallback for extra algorithms beyond the 4 defined above
_EXTRA_COLORS  = ["#7B1FA2", "#1565C0", "#558B2F", "#E65100"]
_EXTRA_MARKERS = ["P", "X", "v", "<"]

THRESHOLD_ALPHA = 0.65
GRID_COLOR      = "#E0E0E0"
BG_COLOR        = "white"

# Fraction of the sweep range used as the minimum gap before jitter is applied
JITTER_THRESHOLD_FRAC = 0.01
# Step size for jitter (fraction of sweep range per slot)
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


# ── Utilities ─────────────────────────────────────────────────────────────────

def state_to_tensor(state_dict):
    """Flatten state dict (sorted keys) into a float32 tensor."""
    kpi_vals, param_vals = [], []
    for key in sorted(state_dict.keys()):
        if "param" in key:
            param_vals.append(state_dict[key])
        elif "kpi" in key:
            kpi_vals.append(state_dict[key])
    return torch.from_numpy(np.concatenate(param_vals + kpi_vals)).float()


def denormalize_params(norm_params, env):
    """Convert normalised parameter values back to raw domain values."""
    raw = np.empty_like(norm_params)
    for i, p in enumerate(env.params):
        lo, hi = p.get_threshold()
        raw[i]  = norm_params[i] * (hi - lo) + lo
    return raw


def algo_style(idx, algo_names):
    """Return the style dict for algorithm index idx."""
    if idx < len(ALGO_STYLES):
        return ALGO_STYLES[idx]
    extra = idx - len(ALGO_STYLES)
    return dict(
        color     = _EXTRA_COLORS[extra % len(_EXTRA_COLORS)],
        marker    = _EXTRA_MARKERS[extra % len(_EXTRA_MARKERS)],
        linestyle = "-",
        label     = algo_names[idx] if idx < len(algo_names) else f"Algo {idx}",
    )


def jitter_values(raw_vals, sweep_range, threshold_frac=JITTER_THRESHOLD_FRAC,
                  step_frac=JITTER_STEP_FRAC):
    """
    Given a list of raw parameter values (one per algorithm), return a list of
    display values with tiny offsets applied to near-identical entries so that
    no two vertical lines overlap on the plot.

    raw_vals     : list[float]  – one value per algorithm (denormalised)
    sweep_range  : float        – thresh_hi - thresh_lo
    Returns      : list[float]  – display values (same order, clamped offsets)
    """
    min_gap  = threshold_frac * sweep_range
    step     = step_frac      * sweep_range
    display  = list(raw_vals)

    # Group by proximity and spread within each group
    indexed  = sorted(enumerate(raw_vals), key=lambda t: t[1])
    i        = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and (indexed[j][1] - indexed[i][1]) < min_gap:
            j += 1
        group = indexed[i:j]
        if len(group) > 1:
            centre = np.mean([v for _, v in group])
            offsets = np.arange(len(group)) * step
            offsets -= offsets.mean()
            for k, (orig_idx, _) in enumerate(group):
                display[orig_idx] = centre + offsets[k]
        i = j

    return display


# ── Conflict-edge detection ───────────────────────────────────────────────────

def detect_conflict_edges(causal_graph, env):
    """
    Scan the CDL binary causal graph for edges  kpi_node → param_id.
    An edge exists when causal_graph[kpi_node, param_id] == 1.

    Returns list of dicts:
        primary_xapp_id : int          – xApp whose KPI drives this edge
        param_id        : int          – parameter index
        xapps_in_conflict : list[xApp] – all xApps that share this parameter
        conflict_xapp_ids : list[int]  – their integer indices
    """
    num_params  = env.num_params
    state_dim   = causal_graph.shape[0]
    param2xapp  = {}                          # param_id → list[xApp]
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
            conflict_xapp_ids = sorted({xapp2id[xa] for xa in xapps_in_conflict}
                                       | {primary_xapp_id})
            edges.append(dict(
                primary_xapp_id   = primary_xapp_id,
                param_id          = param_id,
                xapps_in_conflict = xapps_in_conflict,
                conflict_xapp_ids = conflict_xapp_ids,
            ))
    return edges


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(ax, sweep, curves, algo_actions, algo_names,
               param_id, primary_xapp_id, thresh, KPI_THRESHOLDS, MEAN_STD_KPIS):
    """
    Draw one utility-curve panel.

    sweep        : np.ndarray  – x-axis values (denormalised param range)
    curves       : list[(xapp_id, np.ndarray)]  – (id, utility values)
    algo_actions : list[float] – denormalised param value chosen by each algo
    algo_names   : list[str]
    thresh       : (lo, hi)    – axis limits
    """
    sweep_range = thresh[1] - thresh[0]

    # ── 1. Utility curves ──────────────────────────────────────────────────
    for xapp_id, values in curves:
        color = XAPP_COLORS[xapp_id % len(XAPP_COLORS)]
        lw    = 2.2 if xapp_id == primary_xapp_id else 1.5
        ax.plot(sweep, values, color=color, linewidth=lw, zorder=3)

    # ── 2. KPI threshold horizontal lines (q_i, dashed) ───────────────────
    for xapp_id, _ in curves:
        if xapp_id >= len(KPI_THRESHOLDS):
            continue
        raw_thresh = KPI_THRESHOLDS[xapp_id]
        mean, std  = MEAN_STD_KPIS[xapp_id]
        q_i        = (raw_thresh - mean) / std
        color      = XAPP_COLORS[xapp_id % len(XAPP_COLORS)]
        ax.axhline(q_i, color=color, linewidth=1.2, linestyle="--",
                   alpha=THRESHOLD_ALPHA, zorder=2)

    # ── 3. η* peak line of the primary xApp curve ─────────────────────────
    primary_vals = next(v for xid, v in curves if xid == primary_xapp_id)
    peak_idx     = int(np.argmax(primary_vals))
    eta_val      = float(primary_vals[peak_idx])
    ax.axhline(eta_val, color="#9E9E9E", linewidth=1.0,
               linestyle="--", alpha=0.8, zorder=2,
               label=r"$\eta^{U^*}$")
    ax.scatter([sweep[peak_idx]], [eta_val],
               color="#F44336", s=55, zorder=6, linewidths=0)

    # ── 4. Algorithm vertical lines (with jitter for overlaps) ────────────
    #  Clamp values to axis limits so axvline never silently disappears
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

        # Small text label just above the x-axis showing the raw value
        ax.text(disp_val, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else
                ax.dataLim.y0,
                f" {raw_val:.1f}", color=style["color"],
                fontsize=7, va="bottom", ha="center", zorder=8,
                rotation=90, clip_on=True)

    # ── 5. Axes ────────────────────────────────────────────────────────────
    ax.set_xlim(thresh[0], thresh[1])
    ax.set_xlabel(f"Values for $p_{{{param_id}}}$", labelpad=4)
    ax.set_ylabel(f"Values for $U(p_{{{param_id}}})$", labelpad=4)

    all_ids    = [xid for xid, _ in curves]
    others     = [xid for xid in all_ids if xid != primary_xapp_id]
    others_str = ", ".join(f"$x_{{{i}}}$" for i in others)
    title      = (f"$x_{{{primary_xapp_id}}}$ conflict with {others_str} "
                  f"over $p_{{{param_id}}}$") if others else \
                 f"$x_{{{primary_xapp_id}}}$ over $p_{{{param_id}}}$"
    ax.set_title(title, pad=7)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env       = get_env()
    state_dim = env.get_state_dim()
    act_dim   = env.get_action_dim()
    model     = get_model(env)

    # Load CDL
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

    # Binary causal graph  [state_dim × num_params]
    causal_graph = cdl.get_binary_graph()[:, :-1].cpu().detach().numpy()

    algorithms   = get_algorithms(model, env)
    algo_names   = [a.name for a in algorithms]
    utility_fns  = env.get_utility_fns()

    # ── Step 1: Detect conflict edges ──────────────────────────────────────
    edges = detect_conflict_edges(causal_graph, env)

    if not edges:
        print("[WARN] No conflict edges found in causal graph.")
        return 0

    print(f"Detected {len(edges)} conflict edge(s).\n")

    # ── Step 2 & 3: Freeze one shared state, then for each conflict edge
    #               give every algorithm the exact same tensor ──────────────
    #
    # The state is captured once after a single env.reset() and stored as
    # an immutable tensor.  Before each algo.act() call we hand the algo
    # that same tensor — we never call env.reset() or env.step() again
    # inside the loop, so no algorithm can see a mutated state.
    #
    # If algo.act() needs an env reference and calls env methods internally,
    # we also snapshot raw_params from this frozen state so that the utility
    # sweep is always computed relative to the same baseline.

    env.reset()
    # Capture a single, frozen snapshot — clone() makes it truly independent
    frozen_state_t = state_to_tensor(env._get_state()).to(DEVICE)
    frozen_state_t = frozen_state_t.clone()          # safeguard against in-place ops
    raw_params     = denormalize_params(
        frozen_state_t[:env.num_params].cpu().numpy(), env
    )

    print(f"Frozen evaluation state (params): "
          f"{raw_params.round(3).tolist()}\n")
    
    KPI_THRESHOLDS, MEAN_STD_KPIS=env.get_thresholds_stds()

    panels = []
    for edge in edges:
        param_id          = edge["param_id"]
        primary_xapp_id   = edge["primary_xapp_id"]
        xapps_in_conflict = edge["xapps_in_conflict"]
        conflict_xapp_ids = edge["conflict_xapp_ids"]

        param  = env.params[param_id]
        thresh = param.get_threshold()
        sweep  = np.linspace(thresh[0], thresh[1], 300)

        # Every algorithm receives an identical clone of the frozen tensor.
        # clone() + detach() prevents any algorithm from accidentally
        # accumulating gradients or modifying the shared buffer.
        algo_actions = []
        for algo in algorithms:
            state_for_algo = frozen_state_t.clone().detach()
            
            # Compute weights based on conflict severity
            # Initialize with equal weights, then adjust based on primary xApp priority
            num_xapps = len(xapps_in_conflict)
            weights = np.ones(num_xapps)
            
            # Increase weight for primary xApp (the one driving the conflict)
            # This ensures the algorithm prioritizes satisfying the primary xApp's KPI
            for idx, xapp in enumerate(xapps_in_conflict):
                if xapp == env.xapps[primary_xapp_id]:
                    weights[idx] *= 2.0  # Primary xApp gets 2x weight
            
            # Normalize weights to sum to num_xapps
            weights = weights / weights.sum() * num_xapps
            
            action = algo.act(
                current_state          = state_for_algo,
                conflict_param_index   = param_id,
                xapps_under_conflict   = xapps_in_conflict,
                weights_per_xapps      = weights.tolist(),
                scaling_term           = 10,
            )
            raw_val = env.action_to_param(action)[1]
            algo_actions.append(raw_val)

        # Compute utility curves for every involved xApp across the sweep
        curves = []
        for xid in conflict_xapp_ids:
            if xid >= len(utility_fns):
                continue
            values = []
            for v in sweep:
                p = raw_params.copy()
                p[param_id] = v
                values.append(utility_fns[xid](p))
            curves.append((xid, np.array(values)))

        panels.append(dict(
            primary_xapp_id = primary_xapp_id,
            param_id        = param_id,
            thresh          = thresh,
            sweep           = sweep,
            curves          = curves,
            algo_actions    = algo_actions,
        ))

        print(f"  [xApp {primary_xapp_id} · param {param_id}]  "
              f"conflict with xApps {conflict_xapp_ids}")
        for name, val in zip(algo_names, algo_actions):
            print(f"    {name:22s} → {val:8.4f}")
    print()

    # ── Step 4: Lay out figure ─────────────────────────────────────────────
    n      = len(panels)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    all_xapp_ids = sorted({xid for p in panels for xid, _ in p["curves"]})

    # Build legend (utility curves + thresholds)
    legend_handles = []
    for xid in all_xapp_ids:
        c = XAPP_COLORS[xid % len(XAPP_COLORS)]
        # Match the primary xApp linewidth (2.2) used in draw_panel
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2.2, label=f"$k_{{{xid}}}$"))
    for xid in all_xapp_ids:
        if xid >= len(KPI_THRESHOLDS):
            continue
        c = XAPP_COLORS[xid % len(XAPP_COLORS)]
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=1.2, linestyle="--",
                   alpha=THRESHOLD_ALPHA, label=f"$q_{{{xid}}}$"))
    legend_handles.append(
        Line2D([0], [0], color="#9E9E9E", linewidth=1.0,
               linestyle="--", label=r"$\eta^{U^*}$"))
    legend_handles.append(Line2D([0], [0], color="none", label=""))  # spacer
    for idx, name in enumerate(algo_names):
        s = algo_style(idx, algo_names)
        legend_handles.append(
            Line2D([0], [0], color=s["color"], linewidth=1.6,
                   linestyle=s["linestyle"],
                   marker=s["marker"] or "None", markersize=6,
                   label=name))

    legend_h = 1.0
    panel_h  = 4.2
    title_h  = 0.5
    fig_h    = panel_h * n_rows + title_h + legend_h
    leg_frac = legend_h / fig_h

    fig = plt.figure(figsize=(5.8 * n_cols, fig_h), facecolor=BG_COLOR)
    fig.suptitle(
        "xApp Utility Curves  ·  Algorithm Parameter Selections",
        fontsize=13, fontweight="bold", color="#212121",
        y=1.0 - 0.08 / fig_h,
    )

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig, hspace=0.65, wspace=0.40,
        left=0.07, right=0.97,
        top=1.0 - (title_h / fig_h),
        bottom=leg_frac + 0.02,
    )

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax   = fig.add_subplot(gs[r, c])
        draw_panel(
            ax              = ax,
            sweep           = panel["sweep"],
            curves          = panel["curves"],
            algo_actions    = panel["algo_actions"],
            algo_names      = algo_names,
            param_id        = panel["param_id"],
            primary_xapp_id = panel["primary_xapp_id"],
            thresh          = panel["thresh"],
            KPI_THRESHOLDS   = KPI_THRESHOLDS,
            MEAN_STD_KPIS   = MEAN_STD_KPIS
        )

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        fig.add_subplot(gs[r, c]).set_visible(False)

    # Single global legend at bottom
    fig.legend(
        handles       = legend_handles,
        loc           = "lower center",
        bbox_to_anchor= (0.5, 0.0),
        ncol          = min(len(legend_handles), 7),
        fontsize      = 8.5,
        frameon       = True,
        framealpha    = 0.95,
        edgecolor     = "#BDBDBD",
        facecolor     = BG_COLOR,
        handlelength  = 1.8,
        handletextpad = 0.5,
        columnspacing = 1.0,
        borderpad     = 0.6,
        labelspacing  = 0.4,
    )

    fname = f"evaluate_output_{'CMI' if USE_CMI else 'MLP'}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved → {fname}")
    plt.show()
    return 0


if __name__ == "__main__":
    main()