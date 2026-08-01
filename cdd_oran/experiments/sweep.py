import logging
from dataclasses import replace
from pathlib import Path

from cdd_oran.config import load_config
from cdd_oran.experiments.evaluate import main as evaluate
from cdd_oran.experiments.train import main as train
from cdd_oran.utils.sweeps import aggregate as aggregate_sweeps
from cdd_oran.utils.sweeps import append_run, graph_run_for_seed

logger = logging.getLogger(__name__)


def run_sweep(config_path, seeds, overrides=(), tag=None, graph_sweep=None):
    seeds = [int(seed) for seed in seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique comma-separated integers")
    cfg = load_config(config_path, overrides)
    tag = tag or Path(config_path).stem
    sweep_dir = Path("sweeps") / tag

    for seed in seeds:
        run_cfg = replace(cfg, seed=seed)
        run_dir = train(run_cfg)
        graph_run = None
        graph_cfg = None
        if run_cfg.model_kind == "mlp":
            if not graph_sweep:
                raise ValueError("MLP sweeps require --graph-sweep")
            graph_run = graph_run_for_seed(graph_sweep, seed)
            graph_cfg = load_config(Path(graph_run) / "config.yaml")
        evaluate(run_cfg, run_dir=run_dir, graph_run=graph_run, graph_cfg=graph_cfg)
        append_run(sweep_dir, run_dir, run_cfg)
        logger.info("Sweep %s completed seed %d: %s", tag, seed, run_dir)


def run_aggregate(sweep_dirs, output_dir=None):
    output_dir = output_dir or Path("results") / Path(sweep_dirs[0]).name
    return aggregate_sweeps(sweep_dirs, output_dir)
