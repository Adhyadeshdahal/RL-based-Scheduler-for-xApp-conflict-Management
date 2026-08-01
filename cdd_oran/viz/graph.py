import matplotlib.pyplot as plt
import networkx as nx

from cdd_oran.models.base import CausalModel


def visualize_causal_graph(model: CausalModel, threshold=None):
    graph = model.get_binary_graph(threshold=threshold).cpu().numpy()
    fd = graph.shape[0]
    graph_view = nx.DiGraph()
    graph_view.add_nodes_from(range(fd))

    for i in range(fd):
        for j in range(fd):
            if graph[i, j]:
                graph_view.add_edge(j, i)

    ncp_nodes = list(range(model.kpi_start))
    kpi_nodes = list(range(model.kpi_start, fd))

    n_ncp = len(ncp_nodes)
    n_kpi = len(kpi_nodes)

    pos = {}
    for idx, node in enumerate(ncp_nodes):
        pos[node] = (idx * 2.0 / max(n_ncp - 1, 1), 1.0)
    for idx, node in enumerate(kpi_nodes):
        pos[node] = (idx * 2.0 / max(n_kpi - 1, 1), 0.0)

    labels = {i: model.node_names[i] for i in range(fd)}

    ncp_to_kpi_edges = [(u, v) for u, v in graph_view.edges() if u in ncp_nodes and v in kpi_nodes]
    kpi_to_kpi_edges = [(u, v) for u, v in graph_view.edges() if u in kpi_nodes and v in kpi_nodes]
    ncp_to_ncp_edges = [(u, v) for u, v in graph_view.edges() if u in ncp_nodes and v in ncp_nodes]

    plt.figure(figsize=(12, 6))

    nx.draw_networkx_nodes(
        graph_view,
        pos,
        nodelist=ncp_nodes,
        node_color="#AED6F1",
        node_size=1200,
        edgecolors="#2E86C1",
        linewidths=2,
    )
    nx.draw_networkx_nodes(
        graph_view,
        pos,
        nodelist=kpi_nodes,
        node_color="#FADBD8",
        node_size=1200,
        edgecolors="#E74C3C",
        linewidths=2,
    )

    nx.draw_networkx_labels(graph_view, pos, labels=labels, font_size=10)

    nx.draw_networkx_edges(
        graph_view, pos, edgelist=ncp_to_kpi_edges, edge_color="gray", arrows=True, arrowsize=15
    )
    nx.draw_networkx_edges(
        graph_view,
        pos,
        edgelist=kpi_to_kpi_edges,
        edge_color="#A569BD",
        arrows=True,
        arrowsize=15,
        style="dashed",
        connectionstyle="arc3,rad=0.3",
    )
    nx.draw_networkx_edges(
        graph_view,
        pos,
        edgelist=ncp_to_ncp_edges,
        edge_color="#2E86C1",
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.3",
    )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#AED6F1",
            markeredgecolor="#2E86C1",
            markersize=12,
            label="NCP (Control)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FADBD8",
            markeredgecolor="#E74C3C",
            markersize=12,
            label="KPI (Metric)",
        ),
        plt.Line2D([0], [0], color="gray", label="NCP→KPI"),
        plt.Line2D([0], [0], color="#A569BD", linestyle="dashed", label="KPI→KPI Implicit"),
        plt.Line2D([0], [0], color="#2E86C1", label="NCP→NCP"),
    ]
    plt.legend(handles=legend_elements, loc="lower center", ncol=5, frameon=True)

    plt.title("Causal Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
