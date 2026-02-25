import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def get_ground_truth_adjacency():
    D = 11
    gt = np.zeros((D, D))

    # Param self edges
    for i in range(7):
        gt[i, i] = 1

    # Param → KPI
    gt[0, 7] = 1  # P1 → K1
    gt[1, 7] = 1  # P2 → K1

    gt[0, 8] = 1  # P1 → K2
    gt[2, 8] = 1  # P3 → K2

    gt[3, 9] = 1  # P4 → K3
    gt[4, 9] = 1  # P5 → K3

    gt[6,10] = 1  # P7 → K4
    gt[5,10] = 1  # P6 → K4

    # KPI → KPI temporal
    gt[7, 9] = 1  # K1 → K3
    gt[8,10] = 1  # K2 → K4

    return gt


def graph_accuracy(pred, gt):
    total = pred.size
    correct = (pred == gt).sum()
    return correct / total


def extract_conflicts(graph):
    param_idx = list(range(7))
    kpi_idx = list(range(7, 11))

    indirect = []
    implicit = []

    for i in param_idx:
        for j in kpi_idx:
            if graph[i, j] == 1:
                indirect.append((i, j))

    for i in param_idx:
        for j in param_idx:
            if i != j and graph[i, j] == 1:
                implicit.append((i, j))

    return indirect, implicit


import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(adj):
    G = nx.DiGraph()
    
    labels = ["P1","P2","P3","P4","P5","P6","P7","K1","K2","K3","K4"]
    p_nodes = [n for n in labels if n.startswith('P')]
    k_nodes = [n for n in labels if n.startswith('K')]
    
    G.add_nodes_from(labels)
    
    indirect, implicit = extract_conflicts(adj)

    for i, j in indirect:
        G.add_edge(labels[i], labels[j], color='red', weight=1.5, type='Indirect')
    # for i, j in implicit:
    #     G.add_edge(labels[i], labels[j], color='blue', weight=1.0, type='Implicit')

    pos = nx.shell_layout(G, [k_nodes, p_nodes])

    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="#A0CBE2", edgecolors="grey")
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    edges = G.edges(data=True)
    for u, v, d in edges:
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=[(u, v)],
            edge_color=d['color'],
            width=d['weight'],
            arrowsize=15,
            connectionstyle="arc3,rad=0.2", 
            alpha=0.6
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Indirect Conflict'),
        Line2D([0], [0], color='blue', lw=2, label='Implicit Conflict')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Conflict Dependency Graph", pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
# Adjacency matrix for P1-P7 and K1-K4
    graph = [
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], # P1
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], # P2
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0], # P3
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0], # P4
        [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0], # P5
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1], # P6
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], # P7
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # K1
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # K2
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # K3
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]  # K4
    ]

    graph = np.array(graph)
    gt = get_ground_truth_adjacency()
    print("Ground Truth Adjacency Matrix:")
    print(gt)

    acc = graph_accuracy(graph, gt)
    print(f"Graph Accuracy: {acc:.4f}")


    visualize_graph(graph)