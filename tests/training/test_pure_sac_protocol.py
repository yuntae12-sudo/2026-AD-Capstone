import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pure_sac_protocol import (
    expected_updates,
    protocol_metadata,
    pure_update,
    require_synthetic_report,
    validate_pure_state,
    validate_pure_evaluation_checkpoint,
    validate_rule_baseline,
)
from scripts.evaluate_sac_models import compare
from tests.training.test_sac_experiment import evaluation


def test_warmup_boundary_updates_once_without_backfilling(monkeypatch):
    from src.training.sac import learner

    calls = []

    def update(state, batch, size):
        calls.append((batch, size))
        return SimpleNamespace(bc_updates=0, updates=state.updates + 1), {"finite": 1}

    monkeypatch.setattr(learner, "update", update)
    replay = SimpleNamespace(capacity=50000, size=4999, sample=lambda n: n)
    state = SimpleNamespace(bc_updates=0, updates=0)
    for count in (4999, 5000, 5001, 5002):
        replay.size = count
        state, metrics = pure_update(state, replay, count)
        assert state.updates == expected_updates(count)
        assert (metrics is None) == (count <= 5000)
    assert calls == [(256, 5001), (256, 5002)]
    with pytest.raises(ValueError, match="counter"):
        pure_update(state, replay, 5002)  # Cannot repeat an already committed update.


def test_pure_recovery_rejects_prefill_bc_and_incorrect_counters():
    replay = SimpleNamespace(
        capacity=50000, size=5001, arrays={"teacher_accepted": np.zeros(5001, bool)}
    )
    state = SimpleNamespace(bc_updates=0, updates=1)
    validate_pure_state(state, replay, 5001)
    state.bc_updates = 5000
    with pytest.raises(ValueError, match="BC"):
        validate_pure_state(state, replay, 5001)
    state.bc_updates = 0
    replay.arrays["teacher_accepted"][15] = True
    with pytest.raises(ValueError, match="teacher"):
        validate_pure_state(state, replay, 5001)
    replay.arrays["teacher_accepted"][15] = False
    replay.size = 5000
    with pytest.raises(ValueError, match="own transitions"):
        validate_pure_state(state, replay, 5001)
    replay.size = 5001
    with pytest.raises(ValueError, match="counter"):
        validate_pure_state(state, replay, 5002)


def test_synthetic_entry_requires_exact_cases_and_full_sampling():
    from scripts.validate_sac_exploration import CASES

    report = dict(
        versions={"v": 1},
        runner_sha256="runner",
        scope="SYNTHETIC_KINEMATICS",
        status="COMPLETE",
        qualified=True,
        passed=True,
        requested_samples=4096,
        cases=[
            dict(
                case=f"v{v}_a{a}_k{k}",
                sigma=sigma,
                mean_valid=True,
                samples=4096,
                accepted=3892,
            )
            for v, a, k in CASES
            for sigma in (0.02, 0.03)
        ],
    )
    require_synthetic_report(report, {"v": 1}, "runner")
    for field, value in (
        ("qualified", False),
        ("versions", {"v": 2}),
        ("runner_sha256", "old"),
    ):
        bad = copy.deepcopy(report)
        bad[field] = value
        with pytest.raises(ValueError, match="synthetic"):
            require_synthetic_report(bad, {"v": 1}, "runner")
    for changes in (
        {"case": "invented"},
        {"samples": 16},
        {"accepted": 3891},
        {"mean_valid": False},
    ):
        bad = copy.deepcopy(report)
        bad["cases"][0].update(changes)
        with pytest.raises(ValueError, match="synthetic"):
            require_synthetic_report(bad, {"v": 1}, "runner")


def test_pure_main_comparison_requires_training_provenance_and_all_seeds():
    reports = [evaluation("teacher", 0)]
    for seed in (0, 1, 2):
        report = evaluation("PURE_SAC", seed)
        report["checkpoint_sha256"] = f"checkpoint-{seed}"
        report["training_evidence"] = dict(
            protocol=protocol_metadata(),
            bc_updates=0,
            initial_replay_size=0,
            updates=10000,
            checkpoint_sha256=f"checkpoint-{seed}",
        )
        reports.append(report)
    compared = compare(reports, "pure-sac")
    assert compared["by_behavior"]["KEEP"]["collision_rate"][
        "pure_sac_minus_rule_based_by_seed"
    ] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="seeds"):
        compare(reports[:-1], "pure-sac")
    with pytest.raises(ValueError, match="BC"):
        compare(reports + [evaluation("BC", 0)], "pure-sac")
    reports[-1]["training_evidence"]["initial_replay_size"] = 20000
    with pytest.raises(ValueError, match="provenance"):
        compare(reports, "pure-sac")


def test_bc_checkpoint_cannot_be_relabelled_as_pure_sac():
    state = SimpleNamespace(bc_updates=0, updates=10000)
    run = dict(
        phase="ONLINE_SAC",
        report=dict(
            training_mode="pure-sac",
            protocol=protocol_metadata(),
            initial_replay_size=0,
            bc_checkpoint_sha256=None,
            valid_online_transitions=15000,
        ),
    )
    validate_pure_evaluation_checkpoint(state, run)
    for field, value in (
        ("training_mode", "bc-sac"),
        ("bc_checkpoint_sha256", "teacherweights"),
        ("valid_online_transitions", 14999),
    ):
        bad = copy.deepcopy(run)
        bad["report"][field] = value
        with pytest.raises(ValueError, match="pure SAC"):
            validate_pure_evaluation_checkpoint(state, bad)
    state.bc_updates = 5000
    with pytest.raises(ValueError, match="pure SAC"):
        validate_pure_evaluation_checkpoint(state, run)


def test_cli_rejects_bc_input_before_loading_data_or_creating_output(tmp_path):
    output = tmp_path / "forbidden"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_sac_online.py"),
            "--mode",
            "pure-sac",
            "--bc",
            "unused",
            "--manifest",
            "unused",
            "--synthetic",
            "unused",
            "--output",
            str(output),
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "forbids BC" in result.stderr
    assert not output.exists()


def test_cached_rule_baseline_requires_all_scenes_and_unchanged_runtime():
    report = evaluation("teacher", 0)
    expected = [("scene", "fixed_keep"), ("scene", "fixed_follow")]
    validate_rule_baseline(report, {"v": 1}, expected)
    with pytest.raises(ValueError, match="runtime"):
        validate_rule_baseline(report, {"v": 2}, expected)
    with pytest.raises(ValueError, match="coverage"):
        validate_rule_baseline(report, {"v": 1}, expected + [("missing", "fixed_keep")])
    report["by_behavior"]["KEEP"]["success_rate"] = 1.0
    with pytest.raises(ValueError, match="runtime"):
        validate_rule_baseline(report, {"v": 1}, expected)


def test_cold_actor_can_take_finite_sac_update_without_bc():
    import jax
    from src.training.sac.learner import initialize, update_checked
    from tests.training.test_sac_learning import observation

    obs = observation()
    state = initialize(obs, seed=0)
    batch = dict(
        observation=obs,
        next_observation=obs,
        raw_u=np.zeros((2, 30, 2), np.float64),
        reward=np.array([0.1, -20.0]),
        delta_ticks=np.ones(2, np.int32),
        bootstrap=np.array([True, False]),
        teacher_accepted=np.zeros(2, bool),
    )
    after, metrics = update_checked(state, batch)
    assert int(after.bc_updates) == 0 and int(after.updates) == 1
    assert all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(state.actor.params),
            jax.tree_util.tree_leaves(after.actor.params),
        )
    )
