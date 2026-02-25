import torch
import numpy as np
import random
from environment import ORANEnvironment
from CDL import CDL
from groundTruth import get_ground_truth_adjacency, graph_accuracy, extract_conflicts, visualize_graph

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STATE_DIM       = 11        # 7 params + 4 KPIs
ACTION_DIM      = 7         # new param values P1..P7

COLLECT_STEPS   = 20_000    # total environment steps to collect
TRAIN_STEPS     = 10_000    # gradient steps
BATCH_SIZE      = 256
VAL_SPLIT       = 0.1       # fraction of buffer used for CMI evaluation
CMI_EVAL_EVERY  = 500       # evaluate & update causal graph every N train steps
LOG_EVERY       = 100       # print loss every N train steps

HIDDEN_DIM      = 64
PRED_HIDDEN     = [64, 32]
LR              = 3e-4
CMI_THRESHOLD   = 0.05
EMA_DECAY       = 0.999


def collect_transitions(n_steps: int):
    env = ORANEnvironment()
    states, next_states, actions = [], [], []

    for _ in range(n_steps):
        s_t, s_tp1 = env.step()
        a_t = s_tp1[:ACTION_DIM]

        states.append(s_t)
        next_states.append(s_tp1)
        actions.append(a_t)

    states      = torch.tensor(np.array(states),      dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    actions     = torch.tensor(np.array(actions),     dtype=torch.float32)

    return states, next_states, actions


def make_s_batch(states, next_states):
    return torch.stack([states, next_states], dim=1)


def train():
    print("=" * 60)
    print(f"  Collecting {COLLECT_STEPS} transitions from ORANEnvironment …")
    print("=" * 60)

    states, next_states, actions = collect_transitions(COLLECT_STEPS)

    n_val   = int(len(states) * VAL_SPLIT)
    n_train = len(states) - n_val

    train_s   = states[:n_train]
    train_ns  = next_states[:n_train]
    train_a   = actions[:n_train]

    val_s     = states[n_train:]
    val_ns    = next_states[n_train:]
    val_a     = actions[n_train:]

    print(f"  Train samples : {n_train}")
    print(f"  Val   samples : {n_val}\n")

    cdl = CDL(
        state_dim     = STATE_DIM,
        action_dim    = ACTION_DIM,
        hidden_dim    = HIDDEN_DIM,
        pred_hidden   = PRED_HIDDEN,
        lr            = LR,
        cmi_threshold = CMI_THRESHOLD,
        ema_decay     = EMA_DECAY,
    )
    print(f"  Device : {cdl.device}\n")

    val_s_batch = make_s_batch(val_s, val_ns).to(cdl.device)
    val_a       = val_a.to(cdl.device)

    print(f"  Training for {TRAIN_STEPS} steps …")
    print("-" * 60)

    for step in range(1, TRAIN_STEPS + 1):

        idx     = torch.randint(0, n_train, (BATCH_SIZE,))
        s_batch = make_s_batch(train_s[idx], train_ns[idx]).to(cdl.device)
        a_batch = train_a[idx].to(cdl.device)

        loss = cdl.train_step(s_batch, a_batch)

        if step % CMI_EVAL_EVERY == 0:
            cdl.evaluate_cmi(val_s_batch, val_a)

        if step % LOG_EVERY == 0:
            graph = cdl.get_causal_graph()
            n_edges = graph.sum().item()
            print(f"  Step {step:>6d} | loss = {loss:.4f} | causal edges = {int(n_edges)}")

    print("\n" + "=" * 60)
    print("  Final causal graph  (rows = s^i_t,  cols = s^j_{{t+1}})")
    print("  State order: P1 P2 P3 P4 P5 P6 P7 K1 K2 K3 K4")
    print("=" * 60)

    cdl.evaluate_cmi(val_s_batch, val_a)
    graph = cdl.get_causal_graph().cpu()

    labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "K1", "K2", "K3", "K4"]
    header = "       " + "  ".join(f"{l:>2}" for l in labels)
    print(header)
    print("       " + "--" * len(labels) * 2)
    for i, row_label in enumerate(labels):
        row = "  ".join(str(graph[i, j].item()) for j in range(STATE_DIM))
        print(f"  {row_label:>2}  |  {row}")

    gt = get_ground_truth_adjacency()
    print("Ground Truth Adjacency Matrix:")
    print(gt)

    graph = np.array(graph)
    acc = graph_accuracy(graph, gt)
    print(f"Graph Accuracy: {acc:.4f}")

    indirect, implicit = extract_conflicts(graph)

    print("Indirect Conflicts (Param → KPI):")
    print(indirect)

    print("Implicit Conflicts (Param → Param):")
    print(implicit)

    visualize_graph(graph)

    return cdl, graph


if __name__ == "__main__":
    cdl, graph = train()