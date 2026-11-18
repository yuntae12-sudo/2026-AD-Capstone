import pytest
from scripts.run_sac_initialization_trial import regression_gate


def rows(outcomes):
    return [
        dict(scene_id=str(i // 2), provider=str(i % 2), outcome=outcome)
        for i, outcome in enumerate(outcomes)
    ]


def test_trial_requires_preserved_successes_and_no_extra_collisions():
    baseline = rows(["ROUTE_GOAL", "ROUTE_GOAL", "COLLISION", "COLLISION"])
    assert regression_gate(baseline, baseline)["passed"]
    assert regression_gate(baseline, rows(["ROUTE_GOAL"] * 4))["passed"]
    assert not regression_gate(
        baseline, rows(["COLLISION", "ROUTE_GOAL", "ROUTE_GOAL", "COLLISION"])
    )["passed"]
    assert not regression_gate(
        baseline, rows(["RECORD_END", "ROUTE_GOAL", "COLLISION", "COLLISION"])
    )["passed"]


def test_trial_rejects_incomplete_duplicate_and_empty_comparisons():
    baseline = rows(["ROUTE_GOAL", "COLLISION"])
    for candidate in [[], baseline[:1], baseline + [baseline[0]]]:
        with pytest.raises(ValueError):
            regression_gate(baseline, candidate)
    with pytest.raises(ValueError, match="successful"):
        regression_gate(rows(["COLLISION"]), rows(["COLLISION"]))
