"""Regenerate the golden fixtures.

Only run when a behaviour change is intended, and review the diff -- regenerating to
make a failing test pass defeats the lock.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import ENV_NAMES, MODEL_KINDS, record  # noqa: E402

GOLDEN_DIR = Path(__file__).parent / "golden"


def main() -> int:
    GOLDEN_DIR.mkdir(exist_ok=True)
    for env_name in ENV_NAMES:
        for kind in MODEL_KINDS:
            data = record(env_name, kind)
            path = GOLDEN_DIR / f"{env_name}_{kind}.json"
            path.write_text(json.dumps(data, indent=2) + "\n")
            print(
                f"wrote {path.name}: {data['num_conflict_edges']} edges, "
                f"{len(data['results'])} planner results"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
