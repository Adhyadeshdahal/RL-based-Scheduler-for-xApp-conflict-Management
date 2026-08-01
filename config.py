from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml


@dataclass(frozen=True)
class ModelConfig:
    lr: float
    cmi_threshold: float
    eval_tau: float
    grad_clip: float
    generative_fc_dims: tuple[int, ...]
    feature_fc_dims: tuple[int, ...]
    batch_size: int


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int
    init_steps: int
    inference_gradient_steps: int
    eval_steps: int
    plot_freq: int
    test_batch_size: int


@dataclass(frozen=True)
class CEMConfig:
    n_candidate: int
    n_top: int
    n_iter: int


@dataclass(frozen=True)
class MPPIConfig:
    n_samples: int
    temperature: float
    noise_sigma: float


@dataclass(frozen=True)
class MCTSConfig:
    n_simulations: int
    ucb_c: float


@dataclass(frozen=True)
class PlannerConfig:
    n_horizon: int
    cem: CEMConfig
    mppi: MPPIConfig
    mcts: MCTSConfig


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    environment: Literal["EnvironmentI", "EnvironmentII"]
    model_kind: Literal["cdl", "mlp"]
    param_ranges: Literal["train", "ood"]
    num_steps: int
    device: str
    model: ModelConfig
    train: TrainConfig
    planner: PlannerConfig


def load_config(path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).parent / "configs" / path
    with path.open() as config_file:
        values = yaml.safe_load(config_file)

    device = values["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ExperimentConfig(
        seed=values["seed"],
        environment=values["environment"],
        model_kind=values["model_kind"],
        param_ranges=values["param_ranges"],
        num_steps=values["num_steps"],
        device=device,
        model=ModelConfig(
            **{
                **values["model"],
                "generative_fc_dims": tuple(values["model"]["generative_fc_dims"]),
                "feature_fc_dims": tuple(values["model"]["feature_fc_dims"]),
            }
        ),
        train=TrainConfig(**values["train"]),
        planner=PlannerConfig(
            n_horizon=values["planner"]["n_horizon"],
            cem=CEMConfig(**values["planner"]["cem"]),
            mppi=MPPIConfig(**values["planner"]["mppi"]),
            mcts=MCTSConfig(**values["planner"]["mcts"]),
        ),
    )


DEFAULT_CONFIG = load_config("env_i_mlp.yaml")
