"""
train.py  —  Connects ORANEnvironment to CDL and learns the causal graph.

State  : 11-dim  (P1..P7, K1..K4)
Action :  7-dim  (new values of P1..P7 set by xApps each step)
"""

import torch
from torch import Tensor
import numpy as np
import random
from Environment import ORANEnvironment,initialize_environment
from Policies import RandomExploration,InterventionPolicy
from Parameters import *    #contains hyperparameters

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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def collect_transitions(n_steps,cmi_matrix:Tensor):
    params,kpis,xapps = initialize_environment()

    policy = RandomExploration(
        params=params,
        kpis=kpis,
        xapps=xapps,
        cmi_threshold=CMI_THRESHOLD,
        epsilon=EPSILON,
        cmi_matrix=cmi_matrix)
    
    env = ORANEnvironment(params=params,kpis=kpis,xapps=xapps,policy=policy)
    states, next_states, actions = [], [], []

    for _ in range(n_steps):
        s_t, s_tp1 = env.step()
        a_t = s_tp1[:ACTION_DIM]
        states.append(s_t)
        next_states.append(s_tp1)
        actions.append(a_t)

    states      = np.array(states,      dtype=np.float32)  
    next_states = np.array(next_states, dtype=np.float32)  
    actions     = np.array(actions,     dtype=np.float32) 

    mean_a = actions.mean(axis=0)                          
    std_a  = actions.std(axis=0) + 1e-8                    

    mean_s = np.concatenate([mean_a, np.zeros(STATE_DIM - ACTION_DIM)])   
    std_s  = np.concatenate([std_a,  np.ones(STATE_DIM  - ACTION_DIM)])   

    actions     = (actions     - mean_a) / std_a
    states      = (states      - mean_s) / std_s
    next_states = (next_states - mean_s) / std_s

    return (
        torch.tensor(states,      dtype=torch.float32),
        torch.tensor(next_states, dtype=torch.float32),
        torch.tensor(actions,     dtype=torch.float32),
    )


def make_s_batch(states, next_states):
    return torch.stack([states, next_states], dim=1)


def train():
    gt = get_ground_truth_adjacency()
    cdl = CDL(
        state_dim     = STATE_DIM,
        action_dim    = ACTION_DIM,
        hidden_dim    = HIDDEN_DIM,
        pred_hidden   = PRED_HIDDEN,
        lr            = LR,
        cmi_threshold = CMI_THRESHOLD,
        ema_decay     = EMA_DECAY,
    )

    cmi_matrix = cdl.get_cmi_matrix()

    
    print("=" * 60)
    print(f"  Collecting {COLLECT_STEPS} transitions ...")
    print("=" * 60)
    states, next_states, actions = collect_transitions(COLLECT_STEPS,cmi_matrix)

    n_val   = int(len(states) * VAL_SPLIT)
    n_train = len(states) - n_val
    train_s, train_ns, train_a = states[:n_train], next_states[:n_train], actions[:n_train]
    val_s,   val_ns,   val_a   = states[n_train:], next_states[n_train:], actions[n_train:]

    print(f"  Train: {n_train}  Val: {n_val}\n")

    print(f"  Device: {cdl.device}\n")

    val_s_batch = make_s_batch(val_s, val_ns).to(cdl.device)
    val_a       = val_a.to(cdl.device)

    print(f"  Training for {TRAIN_STEPS} steps ...")
    print("-" * 60)

    for step in range(1, TRAIN_STEPS + 1):
        idx     = torch.randint(0, n_train, (BATCH_SIZE,))
        s_batch = make_s_batch(train_s[idx], train_ns[idx]).to(cdl.device)
        a_batch = train_a[idx].to(cdl.device)
        loss    = cdl.train_step(s_batch, a_batch)

        if step % CMI_EVAL_EVERY == 0:
            cdl.evaluate_cmi(val_s_batch, val_a)

        if step % LOG_EVERY == 0:
            n_edges = cdl.get_causal_graph().sum().item()
            print(f"  Step {step:>6d} | loss = {loss:.4f} | edges = {int(n_edges)}")

    cdl.evaluate_cmi(val_s_batch, val_a)
    graph = cdl.get_causal_graph().cpu().numpy()

    print("\n" + "=" * 60)
    print("  Learned causal graph")
    print("=" * 60)
    header = "       " + "  ".join(f"{l:>2}" for l in LABELS)
    print(header)
    print("       " + "--" * len(LABELS) * 2)
    for i, row_label in enumerate(LABELS):
        row = "  ".join(str(graph[i, j]) for j in range(STATE_DIM))
        print(f"  {row_label:>2}  |  {row}")

    print("\n  Graph metrics vs ground truth:")
    print_metrics(graph, gt)

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
    else:
        print("\n  No spurious Param -> Param edges.")

    plot_cmi_heatmap(cmi_matrix, labels=LABELS,
                     title=f"Normalized CMI (threshold={CMI_THRESHOLD})")
    visualize_graph(graph, title="Learned Causal Graph", labels=LABELS)
    compare_graphs(graph, gt)

    return cdl, graph


if __name__ == "__main__":
    cdl, graph = train()