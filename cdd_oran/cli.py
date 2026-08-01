import argparse
import logging
from dataclasses import replace
from pathlib import Path

from cdd_oran.config import load_config
from cdd_oran.experiments.sweep import run_aggregate, run_sweep
from cdd_oran.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _add_log_level(parser):
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")


def _load_run_config(run_dir, config_path):
    return load_config(config_path or Path(run_dir) / "config.yaml")


def build_parser():
    parser = argparse.ArgumentParser(description="CDD O-RAN experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a world model")
    train.add_argument("--config", default="configs/env_i_mlp.yaml", help="YAML experiment config")
    train.add_argument("--seed", type=int, help="Override the config seed")
    train.add_argument("--resume", action="store_true", help="Resume the checkpoint in --run")
    train.add_argument("--run", help="Existing run directory to resume")
    train.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config path",
    )
    _add_log_level(train)

    evaluate = commands.add_parser("evaluate", help="Evaluate planners using a trained run")
    evaluate.add_argument("--run", required=True, help="Training run containing checkpoint.pt")
    evaluate.add_argument("--config", help="Override config path; defaults to RUN/config.yaml")
    evaluate.add_argument(
        "--graph-run",
        help="Required for MLP runs: trained CDL run supplying the causal graph checkpoint",
    )
    _add_log_level(evaluate)

    viz = commands.add_parser("viz", help="Visualize a CDL run")
    viz.add_argument("--run", required=True, help="CDL training run containing checkpoint.pt")
    viz.add_argument(
        "--figure", choices=("causal-graph", "cmi-heatmap", "panels"), default="causal-graph"
    )
    viz.add_argument("--out", help="Output image path for the panels figure")
    _add_log_level(viz)

    sweep = commands.add_parser("sweep", help="Train and evaluate one run per seed")
    sweep.add_argument("--config", required=True, help="YAML experiment config")
    sweep.add_argument("--seeds", required=True, help="Comma-separated training seeds")
    sweep.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config path",
    )
    sweep.add_argument("--tag", help="Sweep manifest name under sweeps/")
    sweep.add_argument(
        "--graph-sweep",
        help="Required for MLP: manifest containing one trained CDL run for each seed",
    )
    _add_log_level(sweep)

    aggregate_parser = commands.add_parser(
        "aggregate", help="Aggregate utilities across sweep seeds"
    )
    aggregate_parser.add_argument("--sweep", action="append", required=True, help="Sweep directory")
    aggregate_parser.add_argument("--out", help="Output directory (default: results/<sweep-tag>)")
    _add_log_level(aggregate_parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.command == "train":
            if args.resume and not args.run:
                raise ValueError("--resume requires --run")
            cfg = load_config(args.config, args.set)
            if args.seed is not None:
                cfg = replace(cfg, seed=args.seed)
            from cdd_oran.experiments.train import main as train

            train(cfg, resume=args.resume, run_dir=Path(args.run) if args.run else None)
            return 0

        if args.command == "evaluate":
            cfg = _load_run_config(args.run, args.config)
            graph_cfg = _load_run_config(args.graph_run, None) if args.graph_run else None
            from cdd_oran.experiments.evaluate import main as evaluate

            return evaluate(cfg, run_dir=args.run, graph_run=args.graph_run, graph_cfg=graph_cfg)

        if args.command == "sweep":
            run_sweep(args.config, args.seeds, args.set, args.tag, args.graph_sweep)
            return 0

        if args.command == "aggregate":
            output_dir = run_aggregate(args.sweep, args.out)
            logger.info("Aggregate written to %s", output_dir)
            return 0

        cfg = _load_run_config(args.run, None)
        from cdd_oran.experiments.viz import main as visualize

        return visualize(cfg, run_dir=args.run, figure=args.figure, out=args.out)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
