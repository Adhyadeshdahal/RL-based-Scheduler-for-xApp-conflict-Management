import logging
from pathlib import Path

from cdd_oran.config import DEFAULT_CONFIG, ExperimentConfig
from cdd_oran.envs import get_env
from cdd_oran.models import get_model
from cdd_oran.utils.seeding import seed_everything
from cdd_oran.viz import visualize_causal_graph, visualize_cmi_heatmap, visualize_panels

logger = logging.getLogger(__name__)


def main(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    run_dir=None,
    figure="all",
    out=None,
    show=False,
):
    if run_dir is None:
        raise ValueError("An experiment run directory is required")
    if figure not in {"all", "causal-graph", "cmi-heatmap", "panels"}:
        raise ValueError(f"Unsupported figure: {figure}")

    run_path = Path(run_dir)
    figures = ("causal-graph", "cmi-heatmap", "panels") if figure == "all" else (figure,)
    if figure == "all":
        output_dir = Path(out) if out else run_path
        output_paths = {name: output_dir / f"{name}.png" for name in figures}
    else:
        output_paths = {
            figure: Path(out) if out else run_path / f"{figure}.png",
        }

    model = None
    model_error = None
    if cfg.model_kind == "cdl" and any(name != "panels" for name in figures):
        try:
            seed_everything(cfg.seed, cfg.deterministic)
            env = get_env(cfg)
            model = get_model(cfg, env)
            checkpoint_path = run_path / "checkpoint.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            model.load_model(checkpoint_path)
        except (FileNotFoundError, ValueError) as error:
            model_error = str(error)

    generated = 0
    for name in figures:
        output_path = output_paths[name]
        if name in ("causal-graph", "cmi-heatmap"):
            if cfg.model_kind != "cdl":
                logger.info(
                    "Skipped %s: run model_kind is %s; requires a CDL model",
                    name,
                    cfg.model_kind,
                )
                continue
            if model is None:
                logger.info("Skipped %s: causal model unavailable: %s", name, model_error)
                continue
            try:
                if name == "causal-graph":
                    visualize_causal_graph(model, output_path=output_path, show=show)
                else:
                    visualize_cmi_heatmap(model, output_path=output_path, show=show)
            except (FileNotFoundError, ValueError) as error:
                logger.info("Skipped %s: %s", name, error)
                continue
        else:
            utilities_path = run_path / "utilities.json"
            if not utilities_path.exists():
                logger.info("Skipped panels: utilities.json is missing; run has not been evaluated")
                continue
            try:
                visualize_panels(utilities_path, output_path)
            except (FileNotFoundError, ValueError) as error:
                logger.info("Skipped panels: %s", error)
                continue

        generated += 1
        logger.info("Wrote %s", output_path)

    return 0 if generated else 1
