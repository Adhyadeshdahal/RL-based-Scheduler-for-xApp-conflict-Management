import numpy as np
from matplotlib import rcParams

XAPP_COLORS = [
    "#E91E8C",
    "#00BCD4",
    "#4CAF50",
    "#FF9800",
    "#3F51B5",
]

ALGO_STYLES = {
    "QACM": dict(color="#D32F2F", marker="o", linestyle="-", label="QACM"),
    "ModelBasedMCTS": dict(color="#212121", marker="D", linestyle="-", label="ModelBasedMCTS"),
    "ModelBasedMPPI": dict(color="#F44336", marker="^", linestyle="--", label="ModelBasedMPPI"),
    "ModelBasedCEM": dict(color="#00897B", marker="s", linestyle="-.", label="ModelBasedCEM"),
}
_EXTRA_COLORS = ["#7B1FA2", "#1565C0", "#558B2F", "#E65100"]
_EXTRA_MARKERS = ["P", "X", "v", "<"]

THRESHOLD_ALPHA = 0.65
GRID_COLOR = "#E0E0E0"
BG_COLOR = "white"
JITTER_THRESHOLD_FRAC = 0.01
JITTER_STEP_FRAC = 0.015

rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": BG_COLOR,
        "figure.facecolor": BG_COLOR,
        "axes.edgecolor": "#BDBDBD",
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#424242",
        "ytick.color": "#424242",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.labelcolor": "#212121",
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#212121",
    }
)


def algo_style(name):
    if name in ALGO_STYLES:
        return ALGO_STYLES[name]
    extra = len(ALGO_STYLES)
    return dict(
        color=_EXTRA_COLORS[extra % len(_EXTRA_COLORS)],
        marker=_EXTRA_MARKERS[extra % len(_EXTRA_MARKERS)],
        linestyle="-",
        label=name,
    )


def jitter_values(
    raw_vals, sweep_range, threshold_frac=JITTER_THRESHOLD_FRAC, step_frac=JITTER_STEP_FRAC
):
    min_gap = threshold_frac * sweep_range
    step = step_frac * sweep_range
    display = list(raw_vals)
    indexed = sorted(enumerate(raw_vals), key=lambda t: t[1])
    i = 0
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


def draw_panel(
    ax,
    sweep,
    curves,
    algo_actions,
    algo_names,
    param_id,
    primary_xapp_id,
    thresh,
    KPI_THRESHOLDS,
    MEAN_STD_KPIS,
):
    sweep_range = thresh[1] - thresh[0]

    for xapp_id, values in curves:
        color = XAPP_COLORS[xapp_id % len(XAPP_COLORS)]
        lw = 2.2 if xapp_id == primary_xapp_id else 1.5
        ax.plot(sweep, values, color=color, linewidth=lw, zorder=3)

    for xapp_id, _ in curves:
        if xapp_id >= len(KPI_THRESHOLDS):
            continue
        raw_thresh = KPI_THRESHOLDS[xapp_id]
        mean, std = MEAN_STD_KPIS[xapp_id]
        q_i = (raw_thresh - mean) / std
        color = XAPP_COLORS[xapp_id % len(XAPP_COLORS)]
        ax.axhline(q_i, color=color, linewidth=1.2, linestyle="--", alpha=THRESHOLD_ALPHA, zorder=2)

    primary_vals = next(v for xid, v in curves if xid == primary_xapp_id)
    peak_idx = int(np.argmax(primary_vals))
    eta_val = float(primary_vals[peak_idx])
    ax.axhline(eta_val, color="#9E9E9E", linewidth=1.0, linestyle="--", alpha=0.8, zorder=2)
    ax.scatter([sweep[peak_idx]], [eta_val], color="#F44336", s=55, zorder=6, linewidths=0)

    clamped = [float(np.clip(v, thresh[0], thresh[1])) for v in algo_actions]
    display = jitter_values(clamped, sweep_range)

    for idx, (disp_val, raw_val) in enumerate(zip(display, algo_actions, strict=True)):
        style = algo_style(algo_names[idx])
        y_at = float(np.interp(disp_val, sweep, primary_vals))
        ax.axvline(
            disp_val,
            color=style["color"],
            linewidth=1.6,
            linestyle=style["linestyle"],
            alpha=0.92,
            zorder=5,
        )
        if style["marker"]:
            ax.scatter(
                [disp_val],
                [y_at],
                color=style["color"],
                marker=style["marker"],
                s=55,
                zorder=7,
                linewidths=0,
            )
        ax.text(
            disp_val,
            ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else ax.dataLim.y0,
            f" {raw_val:.1f}",
            color=style["color"],
            fontsize=7,
            va="bottom",
            ha="center",
            zorder=8,
            rotation=90,
            clip_on=True,
        )

    ax.set_xlim(thresh[0], thresh[1])
    ax.set_xlabel(f"Values for $p_{{{param_id}}}$", labelpad=4)
    ax.set_ylabel(f"Values for $U(p_{{{param_id}}})$", labelpad=4)
    others = [xid for xid, _ in curves if xid != primary_xapp_id]
    others_str = ", ".join(f"$x_{{{i}}}$" for i in others)
    title = (
        (f"$x_{{{primary_xapp_id}}}$ conflict with {others_str} over $p_{{{param_id}}}$")
        if others
        else f"$x_{{{primary_xapp_id}}}$ over $p_{{{param_id}}}$"
    )
    ax.set_title(title, pad=7)
