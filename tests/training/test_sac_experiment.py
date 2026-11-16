import numpy as np
import pytest
from scripts.sac_experiment_runtime import (
    validate_gate_coverage,
    restore_trace,
    summarize_episodes,
)
from scripts.evaluate_sac_models import compare, BASELINES


def gate():
    episodes = [
        dict(
            scene_id="scene",
            provider=f"fixed_{b.lower()}",
            behavior=b,
            ticks=12,
            accepted=12,
            outcome="COLLISION",
            probe_observations={"0": "a", "10": "b"},
        )
        for b in ("KEEP", "FOLLOW")
    ]
    return dict(
        status="PASSED",
        passed=True,
        probe_ticks=[0, 10, 30],
        episodes=episodes,
        states=[
            dict(scene_id="scene", behavior=b, tick=t)
            for b in ("KEEP", "FOLLOW")
            for t in (0, 10)
        ],
    )


def test_sac_entry_requires_full_validation_and_reachable_probe_coverage():
    report = gate()
    expected = [("scene", "fixed_keep"), ("scene", "fixed_follow")]
    validate_gate_coverage(report, expected)
    with pytest.raises(ValueError, match="coverage"):
        validate_gate_coverage(report, expected + [("omitted", "fixed_keep")])
    report["states"].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_gate_coverage(report, expected)
    report = gate()
    report["episodes"][1]["accepted"] = 10
    with pytest.raises(ValueError, match="mean"):
        validate_gate_coverage(report, expected)
    report = gate()
    report["episodes"][0]["outcome"] = "SMOKE_BATCH_BOUNDARY"
    with pytest.raises(ValueError, match="incomplete"):
        validate_gate_coverage(report, expected)


def test_trace_recovery_never_silently_discards_interrupted_records(tmp_path):
    p = tmp_path / "ticks.jsonl"
    p.write_bytes(b"first\nsecond\n")
    restore_trace(p, 6)
    assert p.read_bytes() == b"first\n"
    assert next(tmp_path.glob("ticks-uncommitted-*.jsonl")).read_bytes() == b"second\n"
    with pytest.raises(ValueError, match="incomplete"):
        restore_trace(p, 100)


def evaluation(policy, seed):
    episodes = [
        dict(
            scene_id="scene",
            provider=f"fixed_{b.lower()}",
            behavior=b,
            ticks=10,
            accepted=5,
            events={"COLLISION": 1},
            sources={"NEW": 5, "EMERGENCY": 5},
            return_training=-20.0,
            stationary_ticks=8,
            tracking_count=0,
            tracking_position_squared=0.0,
            outcome="COLLISION",
        )
        for b in ("KEEP", "FOLLOW")
    ]
    return dict(
        scope="VALIDATION_PERFORMANCE",
        status="COMPLETE",
        qualified=True,
        versions={"v": 1},
        policy=policy,
        seed=seed,
        episodes=episodes,
        by_behavior=summarize_episodes(episodes),
    )


def test_comparison_requires_paired_seeds_and_preserves_failed_policy_metrics():
    reports = [evaluation(p, 0) for p in BASELINES] + [
        evaluation(p, s) for p in ("BC", "SAC") for s in (0, 1, 2)
    ]
    r = compare(reports)
    assert r["by_behavior"]["KEEP"]["collision_rate"]["bc_by_seed"] == [1.0, 1.0, 1.0]
    assert r["by_behavior"]["KEEP"]["stationary_fraction"]["sac_by_seed"] == [
        0.8,
        0.8,
        0.8,
    ]
    with pytest.raises(ValueError, match="seeds"):
        compare(reports[:-1])
    reports[-1]["episodes"].pop()
    with pytest.raises(ValueError, match="unpaired"):
        compare(reports)


def test_braking_baseline_has_no_dependency_on_an_old_plan():
    from tests.environment.test_womd_control import scene
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from scripts.sac_experiment_runtime import scripted_proposer

    e = SACPlanningEnvironment()
    b = e.reset(scene())
    assert e.old_plan is None
    proposal = scripted_proposer("braking_proposal")(e, b.token)
    assert np.isfinite(proposal.raw_u).all()
    assert e.old_plan is None and e.token == b.token
