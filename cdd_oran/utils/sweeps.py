import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from cdd_oran.config import ExperimentConfig, load_config
from cdd_oran.utils.runs import read_metrics


def read_manifest(sweep_dir: str | Path) -> dict[str, Any]:
    path = Path(sweep_dir) / "manifest.json"
    return cast(dict[str, Any], json.loads(path.read_text()))


def append_run(sweep_dir: str | Path, run_dir: str | Path, cfg: ExperimentConfig) -> None:
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    path = sweep_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"runs": []}
    manifest["runs"].append(
        {
            "environment": cfg.environment,
            "model_kind": cfg.model_kind,
            "run_dir": str(run_dir),
            "seed": cfg.seed,
        }
    )
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def graph_run_for_seed(sweep_dir: str | Path, seed: int) -> Path:
    manifest = read_manifest(sweep_dir)
    for run in manifest["runs"]:
        if run["model_kind"] == "cdl" and run["seed"] == seed:
            return Path(run["run_dir"])
    raise ValueError(f"No CDL run for seed {seed} in {sweep_dir}")


def aggregate(sweep_dirs: Iterable[str | Path], output_dir: str | Path) -> Path:
    values: defaultdict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    pooled_values: defaultdict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    planners = set()
    run_keys = set()
    for sweep_dir in sweep_dirs:
        for run in read_manifest(sweep_dir)["runs"]:
            cfg = load_config(Path(run["run_dir"]) / "config.yaml")
            utilities = (
                read_metrics(run["run_dir"]).get("evaluation", {}).get("planner_mean_utilities", {})
            )
            key = (cfg.environment, cfg.model_kind)
            run_keys.add(key)
            planner_values = []
            for planner, value in utilities.items():
                planners.add(planner)
                if value is not None:
                    values[(*key, planner)][run["seed"]] = value
                    planner_values.append(value)
            if planner_values:
                pooled_values[key][run["seed"]] = sum(planner_values) / len(planner_values)

    rows = []
    row_keys = values.keys() | {
        (environment, model_kind, planner)
        for environment, model_kind in run_keys
        for planner in planners
    }
    for environment, model_kind, planner in sorted(row_keys):
        rows.append(
            {
                "environment": environment,
                "model_kind": model_kind,
                "planner": planner,
                **_statistics(values[(environment, model_kind, planner)].values()),
            }
        )

    pooled_rows = [
        {
            "environment": environment,
            "model_kind": model_kind,
            **_statistics(pooled_values[(environment, model_kind)].values()),
        }
        for environment, model_kind in sorted(run_keys)
    ]

    contrasts = _contrasts(values, pooled_values, row_keys, run_keys)
    summary = {"rows": rows, "pooled_rows": pooled_rows, "contrasts": contrasts}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "table.tex").write_text(_table(summary))
    return output_dir


# Two-sided 95% t critical values by degrees of freedom. Sweeps use few seeds, where the
# normal approximation (1.96) understates the interval badly -- t(2) is 4.30.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t95(df: int) -> float:
    return _T95.get(df, 1.96)


def _statistics(values: Iterable[float]) -> dict[str, Any]:
    values = list(values)
    n = len(values)
    if not n:
        return {"mean": None, "std": None, "sem": None, "ci95": None, "n": 0}
    mean = sum(values) / n
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1)) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    half = _t95(n - 1) * sem if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": [mean - half, mean + half],
        "n": n,
    }


def _contrasts(
    values: Any, pooled_values: Any, row_keys: Any, run_keys: Any
) -> list[dict[str, Any]]:
    contrasts = []
    keys = {(environment, planner) for environment, _, planner in row_keys}
    for environment, planner in sorted(keys):
        cdl = values[(environment, "cdl", planner)]
        mlp = values[(environment, "mlp", planner)]
        contrasts.append(_contrast(environment, planner, cdl, mlp))
    for environment in {environment for environment, _ in run_keys}:
        contrasts.append(
            _contrast(
                environment,
                "Pooled",
                pooled_values[(environment, "cdl")],
                pooled_values[(environment, "mlp")],
            )
        )
    return contrasts


def _contrast(environment: str, planner: str, cdl: Any, mlp: Any) -> dict[str, Any]:
    shared_seeds = cdl.keys() & mlp.keys()
    deltas = [cdl[seed] - mlp[seed] for seed in shared_seeds]
    cdl_values = list(cdl.values())
    mlp_values = list(mlp.values())
    if len(cdl_values) > 1 and len(mlp_values) > 1:
        cdl_std = _statistics(cdl_values)["std"]
        mlp_std = _statistics(mlp_values)["std"]
        pooled_std = math.sqrt(
            ((len(cdl_values) - 1) * cdl_std**2 + (len(mlp_values) - 1) * mlp_std**2)
            / (len(cdl_values) + len(mlp_values) - 2)
        )
        cohen_d = (
            (sum(cdl_values) / len(cdl_values) - sum(mlp_values) / len(mlp_values)) / pooled_std
            if pooled_std
            else None
        )
    else:
        cohen_d = None
    return {
        "environment": environment,
        "planner": planner,
        "delta": _statistics(deltas),
        "cohen_d": cohen_d,
        "cdl_n": len(cdl_values),
        "mlp_n": len(mlp_values),
    }


def _table(summary: dict[str, Any]) -> str:
    lines = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Environment & Model & Planner & Mean $\\pm$ SD & 95\\% CI & $n$ \\\\",
        "\\midrule",
    ]
    for row in summary["rows"]:
        lines.append(_table_row(row["environment"], row["model_kind"], row["planner"], row))
    for row in summary["pooled_rows"]:
        lines.append(_table_row(row["environment"], row["model_kind"], "Pooled", row))
    lines.extend(
        ["\\midrule", "Environment & CDL $-$ MLP & Planner & $\\Delta$ & Cohen's $d$ & $n$ \\\\"]
    )
    for contrast in summary["contrasts"]:
        delta = contrast["delta"]
        lines.append(
            f"{contrast['environment']} & CDL $-$ MLP & {contrast['planner']} & "
            f"{_number(delta['mean'])} & {_number(contrast['cohen_d'])} & {delta['n']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _table_row(environment: str, model_kind: str, planner: str, stats: dict[str, Any]) -> str:
    ci = stats["ci95"]
    ci_text = "--" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    return (
        f"{environment} & {model_kind.upper()} & {planner} & "
        f"{_number(stats['mean'])} $\\pm$ {_number(stats['std'])} & {ci_text} & {stats['n']} \\\\"
    )


def _number(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"
