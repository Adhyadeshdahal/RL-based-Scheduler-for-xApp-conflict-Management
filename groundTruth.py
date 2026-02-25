"""
groundtruth.py  —  Ground truth causal graph for ORANEnvironment,
                   evaluation metrics, and visualization utilities.

Node order (matches environment.py):
    indices 0-6  : P1, P2, P3, P4, P5, P6, P7  (params)
    indices 7-10 : K1, K2, K3, K4               (KPIs)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LABELS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "K1", "K2", "K3", "K4"]
PARAM_IDX = list(range(7))
KPI_IDX   = list(range(7, 11))


# ── Ground Truth ──────────────────────────────────────────────────────────────

def get_ground_truth_adjacency():
    """
    Returns the 11x11 ground-truth adjacency matrix derived from environment.py.
    Entry [i, j] = 1  means  s^i_t  causally influences  s^j_{t+1}.

    Causal structure:
        P1 → K1,  P2 → K1          (update_Kpi1)
        P1 → K2,  P3 → K2          (update_Kpi2)
        P4 → K3,  P5 → K3,  K1 → K3   (update_Kpi3)
        P7 → K4,  P6 → K4,  K2 → K4   (update_Kpi4)

    Assumption A3 of CDL: every variable s^i has a self-edge s^i_t → s^i_{t+1}.
    Params are reset each step by xApps (random), so their self-edges are
    technically independent — but we include them per A3.
    """
    D  = 11
    gt = np.zeros((D, D), dtype=int)

    # Self-edges (Assumption A3)
    for i in range(D):
        gt[i, i] = 1

    # Param → KPI edges (from update functions in environment.py)
    gt[0, 7]  = 1   # P1 → K1
    gt[1, 7]  = 1   # P2 → K1
    gt[0, 8]  = 1   # P1 → K2
    gt[2, 8]  = 1   # P3 → K2
    gt[3, 9]  = 1   # P4 → K3
    gt[4, 9]  = 1   # P5 → K3
    gt[5, 10] = 1   # P6 → K4
    gt[6, 10] = 1   # P7 → K4

    # KPI → KPI temporal edges
    gt[7, 9]  = 1   # K1 → K3
    gt[8, 10] = 1   # K2 → K4

    return gt


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def graph_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of correctly predicted edges (both present and absent)."""
    return float((pred == gt).sum()) / pred.size


def graph_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Returns a full set of binary classification metrics for edge prediction.

    TP = edge exists in both pred and gt
    FP = edge in pred but not gt  (spurious edge)
    FN = edge in gt but not pred  (missed edge)
    TN = edge absent in both
    """
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)

    TP = int(( pred_b &  gt_b).sum())
    FP = int(( pred_b & ~gt_b).sum())
    FN = int((~pred_b &  gt_b).sum())
    TN = int((~pred_b & ~gt_b).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / (TP + FP + FN + TN)

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }


def print_metrics(pred: np.ndarray, gt: np.ndarray):
    """Pretty-print all graph metrics."""
    m = graph_metrics(pred, gt)
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}  (low = many spurious edges)")
    print(f"  Recall    : {m['recall']:.4f}  (low = many missed edges)")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")


# ── Edge Classification ───────────────────────────────────────────────────────

def extract_causal_edges(graph: np.ndarray) -> dict:
    """
    Classifies non-self edges in the predicted graph into three groups:

    param_to_kpi  : P_i → K_j  (direct causal influence on a KPI)
    kpi_to_kpi    : K_i → K_j  (temporal KPI dependency)
    param_to_param: P_i → P_j  (unexpected; params are set independently)

    Returns dict with lists of (i, j) index pairs for each group.
    """
    param_to_kpi   = []
    kpi_to_kpi     = []
    param_to_param = []

    for i in range(11):
        for j in range(11):
            if i == j or graph[i, j] == 0:
                continue
            i_is_param = i in PARAM_IDX
            j_is_param = j in PARAM_IDX
            i_is_kpi   = i in KPI_IDX
            j_is_kpi   = j in KPI_IDX

            if i_is_param and j_is_kpi:
                param_to_kpi.append((i, j))
            elif i_is_kpi and j_is_kpi:
                kpi_to_kpi.append((i, j))
            elif i_is_param and j_is_param:
                param_to_param.append((i, j))

    return {
        "param_to_kpi":   param_to_kpi,
        "kpi_to_kpi":     kpi_to_kpi,
        "param_to_param": param_to_param,
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_graph(adj: np.ndarray, title: str = "Causal Dependency Graph",
                    labels: list = None, show_param_param: bool = False):
    """
    Draws the causal graph.
        Red edges    — Param -> KPI
        Purple edges — KPI  -> KPI
        Blue edges   — Param -> Param  (hidden by default)

    Args:
        adj              : square numpy adjacency matrix
        title            : plot title
        labels           : node label list matching adj dimensions.
                           Defaults to full LABELS if not provided.
        show_param_param : whether to draw param->param edges
    """
    if labels is None:
        labels = LABELS

    G = nx.DiGraph()
    G.add_nodes_from(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j or adj[i, j] == 0:
                continue
            i_is_param = labels[i].startswith("P")
            j_is_param = labels[j].startswith("P")
            i_is_kpi   = labels[i].startswith("K")
            j_is_kpi   = labels[j].startswith("K")

            if i_is_param and j_is_kpi:
                G.add_edge(labels[i], labels[j], color="red",       weight=2.0)
            elif i_is_kpi and j_is_kpi:
                G.add_edge(labels[i], labels[j], color="purple",    weight=2.0)
            elif i_is_param and j_is_param and show_param_param:
                G.add_edge(labels[i], labels[j], color="steelblue", weight=1.0)

    p_nodes = [l for l in labels if l.startswith("P")]
    k_nodes = [l for l in labels if l.startswith("K")]
    pos = nx.shell_layout(G, [k_nodes, p_nodes])

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(G, pos, node_size=1400,
                           node_color="#A0CBE2", edgecolors="grey", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=d["color"],
            width=d["weight"],
            arrowsize=18,
            connectionstyle="arc3,rad=0.2",
            alpha=0.75,
        )

    legend_elements = [
        Line2D([0], [0], color="red",      lw=2, label="Param → KPI"),
        Line2D([0], [0], color="purple",   lw=2, label="KPI  → KPI"),
        Line2D([0], [0], color="steelblue",lw=2, label="Param → Param (spurious)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title(title, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def compare_graphs(pred: np.ndarray, gt: np.ndarray):
    """
    Side-by-side plot: ground truth (left) vs CDL prediction (right).
    Green = correct edge, Red = spurious (FP), Orange = missed (FN).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    p_nodes = [l for l in LABELS if l.startswith("P")]
    k_nodes = [l for l in LABELS if l.startswith("K")]

    for ax, adj, title in [(axes[0], gt, "Ground Truth"),
                           (axes[1], pred, "CDL Prediction")]:
        G = nx.DiGraph()
        G.add_nodes_from(LABELS)
        for i in range(11):
            for j in range(11):
                if i != j and adj[i, j] == 1:
                    # colour by correctness when showing prediction
                    if title == "CDL Prediction":
                        if gt[i, j] == 1:
                            color = "green"    # TP
                        else:
                            color = "red"      # FP
                    else:
                        color = "steelblue"
                    G.add_edge(LABELS[i], LABELS[j], color=color)

        # also mark FN in prediction plot
        if title == "CDL Prediction":
            for i in range(11):
                for j in range(11):
                    if i != j and gt[i, j] == 1 and pred[i, j] == 0:
                        G.add_edge(LABELS[i], LABELS[j], color="orange")

        pos = nx.shell_layout(G, [k_nodes, p_nodes])
        plt.sca(ax)
        nx.draw_networkx_nodes(G, pos, node_size=1200,
                               node_color="#A0CBE2", edgecolors="grey", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        edge_colors = [d["color"] for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowsize=15,
                               connectionstyle="arc3,rad=0.2", alpha=0.8, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color="green",      lw=2, label="Correct (TP)"),
        Line2D([0], [0], color="red",        lw=2, label="Spurious (FP)"),
        Line2D([0], [0], color="orange",     lw=2, label="Missed (FN)"),
        Line2D([0], [0], color="steelblue",  lw=2, label="Ground truth edge"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10)
    plt.suptitle("Causal Graph: Ground Truth vs CDL", fontsize=14)
    plt.tight_layout()
    plt.show()




def plot_cmi_heatmap(cmi_matrix, labels: list = None, title: str = "Normalized CMI Matrix"):
    """
    Heatmap of the raw normalized CMI values (before thresholding).
    Useful for choosing the right cmi_threshold — you want it sitting
    between the cluster of strong direct edges and weak indirect ones.

    Args:
        cmi_matrix : (D, D) numpy array or torch tensor
        labels     : node label list
        title      : plot title
    """
    if hasattr(cmi_matrix, "cpu"):
        cmi_matrix = cmi_matrix.cpu().numpy()
    if labels is None:
        labels = LABELS
    D = len(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cmi_matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Normalized CMI")

    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect  s^j_{t+1}")
    ax.set_ylabel("Cause   s^i_t")

    for i in range(D):
        for j in range(D):
            val = cmi_matrix[i, j]
            color = "white" if val > cmi_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title(title, pad=15)
    plt.tight_layout()
    plt.show()

# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    gt = get_ground_truth_adjacency()

    print("Ground Truth Adjacency Matrix:")
    print("  " + "  ".join(f"{l:>2}" for l in LABELS))
    for i, row_label in enumerate(LABELS):
        print(f"{row_label:>2}  " + "   ".join(str(gt[i, j]) for j in range(11)))

    # Sanity check: perfect prediction scores 1.0
    print("\nSelf-check (gt vs gt):")
    print_metrics(gt, gt)

    visualize_graph(gt, title="Ground Truth Causal Graph")