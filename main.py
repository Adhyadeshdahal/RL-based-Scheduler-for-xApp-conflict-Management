"""
train.py  —  Connects ORANEnvironment to CDL and learns the causal graph.

State  : 11-dim  (P1..P7, K1..K4)
Action :  7-dim  (new values of P1..P7 set by xApps each step)
"""

import torch
import numpy as np
import random
from environment import ORANEnvironment
from CDL import CDL
from groundTruth import (
    get_ground_truth_adjacency,
    print_metrics,
    extract_causal_edges,
    visualize_graph,
    compare_graphs,
    plot_cmi_heatmap,
    LABELS,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
STATE_DIM      = 11       # 7 params + 4 KPIs
ACTION_DIM     = 7        # new param values P1..P7

COLLECT_STEPS  = 20_000   # total environment steps to collect
TRAIN_STEPS    = 15_000   # gradient steps
BATCH_SIZE     = 256
VAL_SPLIT      = 0.1      # fraction of buffer used for CMI evaluation
CMI_EVAL_EVERY = 500      # evaluate & update causal graph every N train steps
LOG_EVERY      = 100      # print loss every N train steps

HIDDEN_DIM     = 64
PRED_HIDDEN    = [64, 32]
LR             = 3e-4
CMI_THRESHOLD  = 0.01
EMA_DECAY      = 0.999


# ── Data collection ───────────────────────────────────────────────────────────

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


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    gt = get_ground_truth_adjacency()

    print("=" * 60)
    print(f"  Collecting {COLLECT_STEPS} transitions from ORANEnvironment ...")
    print("=" * 60)

    states, next_states, actions = collect_transitions(COLLECT_STEPS)

    n_val   = int(len(states) * VAL_SPLIT)
    n_train = len(states) - n_val

    train_s  = states[:n_train]
    train_ns = next_states[:n_train]
    train_a  = actions[:n_train]
    val_s    = states[n_train:]
    val_ns   = next_states[n_train:]
    val_a    = actions[n_train:]

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

    print(f"  Training for {TRAIN_STEPS} steps ...")
    print("-" * 60)

    for step in range(1, TRAIN_STEPS + 1):
        idx     = torch.randint(0, n_train, (BATCH_SIZE,))
        s_batch = make_s_batch(train_s[idx], train_ns[idx]).to(cdl.device)
        a_batch = train_a[idx].to(cdl.device)

        loss = cdl.train_step(s_batch, a_batch)

        if step % CMI_EVAL_EVERY == 0:
            cdl.evaluate_cmi(val_s_batch, val_a)

        if step % LOG_EVERY == 0:
            n_edges = cdl.get_causal_graph().sum().item()
            print(f"  Step {step:>6d} | loss = {loss:.4f} | causal edges = {int(n_edges)}")


    # ── Final CMI + abstraction ───────────────────────────────────────────────
    cdl.evaluate_cmi(val_s_batch, val_a)
    cdl.evaluate_action_cmi(val_s_batch, val_a)
    graph = cdl.get_causal_graph().cpu().numpy()

    abstraction = cdl.get_state_abstraction()
    sC = abstraction["sC"]
    sR = abstraction["sR"]
    sI = abstraction["sI"]
    kept        = sC + sR
    kept_labels = [LABELS[i] for i in kept]

    # Pruned adjacency matrix — only sC+sR rows and cols
    pruned = graph[np.ix_(kept, kept)]

    # ── Print full graph ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Full causal graph  (11-dim)")
    print("=" * 60)
    header = "       " + "  ".join(f"{l:>2}" for l in LABELS)
    print(header)
    print("       " + "--" * len(LABELS) * 2)
    for i, row_label in enumerate(LABELS):
        row = "  ".join(str(graph[i, j]) for j in range(STATE_DIM))
        print(f"  {row_label:>2}  |  {row}")

    print("\n  Graph metrics vs ground truth:")
    print_metrics(graph, gt)

    # ── Print pruned graph ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Pruned causal graph  ({len(kept)}-dim, sI dropped)")
    print("=" * 60)
    print(f"  Kept    : {kept_labels}")
    print(f"  Dropped : {[LABELS[i] for i in sI]}")
    header2 = "       " + "  ".join(f"{l:>2}" for l in kept_labels)
    print(header2)
    print("       " + "--" * len(kept_labels) * 2)
    for i, row_label in enumerate(kept_labels):
        row = "  ".join(str(pruned[i, j]) for j in range(len(kept)))
        print(f"  {row_label:>2}  |  {row}")

    # ── Edge summary ──────────────────────────────────────────────────────────
    edges = extract_causal_edges(graph)
    print("\n  Param -> KPI edges:")
    for i, j in edges["param_to_kpi"]:
        print(f"    {LABELS[i]} -> {LABELS[j]}")
    print("\n  KPI -> KPI edges:")
    for i, j in edges["kpi_to_kpi"]:
        print(f"    {LABELS[i]} -> {LABELS[j]}")
    if edges["param_to_param"]:
        print("\n  Spurious Param -> Param edges:")
        for i, j in edges["param_to_param"]:
            print(f"    {LABELS[i]} -> {LABELS[j]}")

    # ── Abstraction summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  State Abstraction  phi(s) = (s^C, s^R)")
    print("=" * 60)
    print(f"  Controllable     (sC): {[LABELS[i] for i in sC]}")
    print(f"  Action-Relevant  (sR): {[LABELS[i] for i in sR]}")
    print(f"  Action-Irrelevant(sI): {[LABELS[i] for i in sI]}  <- pruned")
    print(f"  Abstract state dim  : {len(kept)} / {STATE_DIM}")

    # ── Visualizations ────────────────────────────────────────────────────────
    # 1. Full 11-dim learned graph
    visualize_graph(graph,  title="Full Learned Graph (11-dim)", labels=LABELS)
    # 2. Pruned graph — only sC+sR nodes and their surviving edges
    visualize_graph(pruned, title=f"Pruned Abstract Graph ({len(kept)}-dim, sC+sR only)",
                    labels=kept_labels)
    # 3. Side-by-side comparison with ground truth
    compare_graphs(graph, gt)

    return cdl, graph, abstraction


if __name__ == "__main__":
    cdl, graph, abstraction = train()