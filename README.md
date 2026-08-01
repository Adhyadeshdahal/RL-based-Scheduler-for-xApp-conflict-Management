# CDD O-RAN: Causal Discovery-Driven Model-Based Planning for xApp Conflict Mitigation in O-RAN

This repository contains the framework for resolving configuration conflicts between multiple xApps in the Near-Real-Time RIC of an O-RAN architecture. The framework leverages **Causal Discovery Learning (CDL)** to detect direct, indirect, and implicit conflicts and **Model-Based Planning Algorithms** to dynamically resolve them while optimizing for multiple Key Performance Indicators (KPIs).

![Conflict Resolution Techniques](conflict_resolution_techniques.png)

## Overview

In an Open RAN ecosystem, multiple xApps can concurrently subscribe to and manipulate shared network parameters, leading to configuration conflicts. This framework evaluates several model-based algorithms to find a Pareto-optimal configuration that satisfies the thresholds of all conflicting xApps simultaneously.

### Key Components

- **`Environment/`**: Houses the O-RAN simulation environments.
  - `Environment_I.py`: Replicates a standard 4-xApp scenario from literature (*Toward Control and Coordination in Cognitive Autonomous Networks*).
  - `Environment_II.py`: A more complex 5-xApp, 6-KPI extended environment designed for rigorous algorithm evaluation.
- **`Algorithms/`**: Implementations of various conflict resolution and parameter selection algorithms. Supported planners include:
  - **QACM**: Q-Learning based Action Selection for Conflict Mitigation.
  - **Model-Based CEM**: Cross-Entropy Method for iterative sampling.
  - **Model-Based MPPI**: Model Predictive Path Integral planning using soft-weighted trajectories.
  - **Model-Based MCTS**: Monte Carlo Tree Search for structured exploration.
  *(To add new algorithms, implement the `act` interface and register them in `Algorithms/__init__.py`'s `get_algorithms()` function).*
- **`Models/`**: Contains the learning architectures, primarily the CDL (Causal Discovery Learning) model used to infer the causal dependency graph between O-RAN parameters and KPIs, and the environment transition models.
- **`Parameters/`**: Centralized simulation parameters. Edit `Parameters/__init__.py` to tweak environments, algorithmic hyperparameters, simulation steps, and more.
- **`DATA/`, `Tests/`, `runs/`, `rslts/`**: Directories dedicated to data storage, unit tests, TensorBoard logs, and evaluation results.

## Execution

Use the experiment CLI. Training writes a self-describing run under
`runs/<environment>/<model_kind>/<timestamp>-<confighash>/` with its resolved config,
checkpoint, TensorBoard logs, metrics, and git state.

```bash
uv run python cli.py train --config configs/env_i_cdl.yaml
uv run python cli.py train --config configs/env_i_cdl.yaml --set train.total_steps=100
uv run python cli.py evaluate --run runs/EnvironmentI/cdl/<run-id>
uv run python cli.py viz --run runs/EnvironmentI/cdl/<run-id> --figure cmi-heatmap
```

MLP evaluation requires a trained CDL run for causal conflict detection:

```bash
uv run python cli.py evaluate --run runs/EnvironmentI/mlp/<run-id> \
  --graph-run runs/EnvironmentI/cdl/<cdl-run-id>
```

Run `uv run python cli.py --help` for all options. Use TensorBoard with
`tensorboard --logdir runs/`.

## Setup & Dependencies
This project uses **`uv`** for extremely fast and robust dependency management. 

If you do not have `uv` installed, you can install it using PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Once installed, simply sync the environment:
```bash
uv sync
```

## Visualizing the Causal Graph

Only CDL runs expose a causal graph. Use the `viz` command with either
`causal-graph` or `cmi-heatmap`.
