import torch
import random
import numpy as np

from environment import ORANEnvironment
from CDL import CDL
from groundTruth import get_ground_truth_adjacency, graph_accuracy, extract_conflicts, visualize_graph


def collect_transitions(env, num_steps):
    transitions = []
    for _ in range(num_steps):
        s_t, s_tp1 = env.step()
        transitions.append((s_t, s_tp1))
    return transitions


def create_rollout_dataset(transitions, H):
    data = []
    for i in range(len(transitions) - H):
        rollout = []
        rollout.append(transitions[i][0])
        for h in range(H):
            rollout.append(transitions[i + h][1])
        data.append(rollout)
    return torch.tensor(data, dtype=torch.float32)


def train_val_split(data, val_ratio=0.1):
    N = len(data)
    idx = list(range(N))
    random.shuffle(idx)
    split = int(N * (1 - val_ratio))
    train_idx = idx[:split]
    val_idx = idx[split:]
    return data[train_idx], data[val_idx]


def sample_batch(data, batch_size):
    idx = np.random.choice(len(data), batch_size, replace=False)
    return data[idx]



def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    env = ORANEnvironment()

    num_transitions = 100000
    H = 3
    batch_size = 32
    training_steps = 50000

    transitions = collect_transitions(env, num_transitions)
    dataset = create_rollout_dataset(transitions, H)
    train_data, val_data = train_val_split(dataset)

    state_dim = dataset.shape[-1]

    cdl = CDL(
        state_dim=state_dim,
        hidden_dim=64,
        pred_hidden=[64, 32],
        lr=3e-4,
        H=H,
        cmi_threshold=0.02,
        ema_decay=0.999
    )

    for step in range(training_steps):
        batch = sample_batch(train_data, batch_size)
        loss = cdl.train_step(batch)

        if step % 10 == 0:
            val_batch = sample_batch(val_data, batch_size)
            cdl.evaluate_cmi(val_batch)

        if step % 1000 == 0:
            print(f"Step {step}, Loss {loss:.4f}")

    graph = cdl.get_causal_graph().cpu().numpy()

    print("Learned Adjacency Matrix:")
    print(graph)

    gt = get_ground_truth_adjacency()
    print("Ground Truth Adjacency Matrix:")
    print(gt)

    acc = graph_accuracy(graph, gt)
    print(f"Graph Accuracy: {acc:.4f}")

    indirect, implicit = extract_conflicts(graph)

    print("Indirect Conflicts (Param → KPI):")
    print(indirect)

    print("Implicit Conflicts (Param → Param):")
    print(implicit)

    visualize_graph(graph)



if __name__ == "__main__":
    main()