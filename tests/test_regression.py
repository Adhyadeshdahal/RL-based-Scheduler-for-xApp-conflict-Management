"""Golden regression lock over the numeric core.

A failure means behaviour changed. Understand why before regenerating fixtures with
`uv run python tests/generate_golden.py`. See _harness.py for scope.
"""

import json
from pathlib import Path

import pytest

from tests._harness import ENV_NAMES, MODEL_KINDS, record

GOLDEN_DIR = Path(__file__).parent / "golden"
CASES = [(e, m) for e in ENV_NAMES for m in MODEL_KINDS]
IDS = [f"{e}-{m}" for e, m in CASES]

RTOL = 1e-5
ATOL = 1e-8


def _load(env_name, model_kind):
    path = GOLDEN_DIR / f"{env_name}_{model_kind}.json"
    if not path.exists():
        pytest.fail(f"missing golden fixture {path.name}; run tests/generate_golden.py")
    return json.loads(path.read_text())


@pytest.fixture(scope="module", params=CASES, ids=IDS)
def case(request):
    env_name, model_kind = request.param
    return _load(env_name, model_kind), record(env_name, model_kind)


def test_structure_matches(case):
    """Env shape, ground-truth graph and conflict count are unchanged."""
    golden, actual = case
    for key in (
        "state_dim",
        "action_dim",
        "num_params",
        "true_adj_edges",
        "num_conflict_edges",
    ):
        assert actual[key] == golden[key], f"{key}: {golden[key]} -> {actual[key]}"


def test_initial_state_matches(case):
    """Env construction/reset/observation-noise are still deterministic and identical."""
    golden, actual = case
    assert actual["state"] == pytest.approx(golden["state"], rel=RTOL, abs=ATOL)
    assert actual["raw_params"] == pytest.approx(golden["raw_params"], rel=RTOL, abs=ATOL)


def test_cost_function_matches(case):
    """Continuous probe of the cost fn -- catches drift too small to change an argmin."""
    golden, actual = case
    assert len(actual["cost_probe"]) == len(golden["cost_probe"])
    for g, a in zip(golden["cost_probe"], actual["cost_probe"], strict=True):
        assert a[0] == g[0] and a[1] == g[1], "probe ordering changed"
        assert a[2] == pytest.approx(g[2], rel=RTOL, abs=ATOL), f"xapp{g[0]} u={g[1]}: d"
        assert a[3] == g[3], f"xapp{g[0]} u={g[1]}: satisfaction flag"


def test_model_forward_matches(case):
    """Continuous probe of the world model's forward pass."""
    golden, actual = case
    for g, a in zip(golden["model_probe"], actual["model_probe"], strict=True):
        assert a == pytest.approx(g, rel=RTOL, abs=ATOL)


def test_planner_results_match(case):
    """Every planner returns the same action and utilities on every conflict edge."""
    golden, actual = case
    assert len(actual["results"]) == len(golden["results"])

    mismatches = []
    for g, a in zip(golden["results"], actual["results"], strict=True):
        tag = f"edge{g['edge']} p{g['param_id']} {g['planner']}"
        assert (g["edge"], g["param_id"], g["planner"]) == (
            a["edge"],
            a["param_id"],
            a["planner"],
        ), f"result ordering changed at {tag}"

        if a["action"] != g["action"]:
            mismatches.append(f"{tag}: action {g['action']} -> {a['action']}")
        elif a["raw_val"] != pytest.approx(g["raw_val"], rel=RTOL, abs=ATOL):
            mismatches.append(f"{tag}: raw_val {g['raw_val']} -> {a['raw_val']}")
        elif a["utilities"] != pytest.approx(g["utilities"], rel=RTOL, abs=ATOL):
            mismatches.append(f"{tag}: utilities {g['utilities']} -> {a['utilities']}")

    assert not mismatches, "\n".join(mismatches[:20])


def test_planners_are_not_degenerate(case):
    """Guard the guard: if every planner picked the same action the lock proves nothing."""
    _, actual = case
    by_edge = {}
    for r in actual["results"]:
        by_edge.setdefault(r["edge"], set()).add(tuple(r["action"]))
    assert any(len(v) > 1 for v in by_edge.values()), (
        "all planners agreed on every edge -- fixture is degenerate and would not "
        "detect a planner regression"
    )
