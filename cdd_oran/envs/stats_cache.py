import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "stats"


def get_cached_mean_std(
    environment: str,
    param_ranges: Any,
    seed: int,
    num_samples: int,
    compute: Callable[[], list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    key_data = {
        "environment": environment,
        "param_ranges": param_ranges,
        "seed": seed,
        "num_samples": num_samples,
    }
    key = hashlib.sha256(
        json.dumps(key_data, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    path = _CACHE_DIR / f"{key}.json"

    if path.exists():
        with path.open() as cache_file:
            return [tuple(values) for values in json.load(cache_file)]

    mean_std = compute()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w") as cache_file:
        json.dump(mean_std, cache_file)
    return mean_std
