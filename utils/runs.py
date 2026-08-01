import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from config import config_dict


def create_run_dir(cfg, root="runs"):
    resolved = config_dict(cfg)
    encoded = yaml.safe_dump(resolved, sort_keys=True).encode()
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{hashlib.sha256(encoded).hexdigest()[:8]}"
    run_dir = Path(root) / cfg.environment / cfg.model_kind / run_id
    run_dir.mkdir(parents=True)
    write_metadata(run_dir, cfg)
    return run_dir


def write_metadata(run_dir, cfg):
    run_dir = Path(run_dir)
    with (run_dir / "config.yaml").open("w") as config_file:
        yaml.safe_dump(config_dict(cfg), config_file, sort_keys=False)

    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    (run_dir / "git_sha.txt").write_text(f"{sha}\ndirty={dirty}\n")


def write_metrics(run_dir, metrics):
    with (Path(run_dir) / "metrics.json").open("w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)


def read_metrics(run_dir):
    path = Path(run_dir) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"
