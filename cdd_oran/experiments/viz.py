import logging
from pathlib import Path

from cdd_oran.config import DEFAULT_CONFIG, ExperimentConfig
from cdd_oran.envs import get_env
from cdd_oran.models import get_model
from cdd_oran.utils.seeding import seed_everything
from cdd_oran.viz import visualize_causal_graph, visualize_cmi_heatmap

logger = logging.getLogger(__name__)


def main(cfg: ExperimentConfig = DEFAULT_CONFIG, run_dir=None, figure="causal-graph"):
    if cfg.model_kind == "mlp":
        raise ValueError("MLP runs do not contain an explicit causal graph")
    if run_dir is None:
        raise ValueError("An experiment run directory is required")

    seed_everything(cfg.seed, cfg.deterministic)
    env = get_env(cfg)
    model = get_model(cfg, env)
    checkpoint_path = Path(run_dir) / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_model(checkpoint_path)

    if figure == "causal-graph":
        visualize_causal_graph(model)
    elif figure == "cmi-heatmap":
        visualize_cmi_heatmap(model)
    else:
        raise ValueError(f"Unsupported figure: {figure}")
    logger.info("Displayed %s for %s", figure, run_dir)
    return 0
