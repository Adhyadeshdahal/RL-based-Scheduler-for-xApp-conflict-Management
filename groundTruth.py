import networkx as nx
import matplotlib.pyplot as plt


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


def visualize_graph(adj):
    G = nx.DiGraph()

    labels = [
        "P1","P2","P3","P4","P5","P6","P7",
        "K1","K2","K3","K4"
    ]

    for i in range(len(labels)):
        G.add_node(labels[i])

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] == 1:
                G.add_edge(labels[i], labels[j])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000,
            node_color="lightblue", arrowsize=20)
    plt.show()