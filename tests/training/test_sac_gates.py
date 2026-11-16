import pytest
from src.training.sac.gates import summarize_acceptance, require_training_gate


def test_behavior_specific_gate_cannot_hide_failed_means_or_weak_follow():
    rows = []
    for behavior, mean_count, accepted in (("KEEP", 10, 256), ("FOLLOW", 8, 256)):
        rows.extend(
            dict(
                behavior=behavior,
                mean_pass=i < mean_count,
                samples=256 if i < mean_count else 0,
                accepted=accepted if i < mean_count else 0,
            )
            for i in range(10)
        )
    report = summarize_acceptance(rows)
    assert report["by_behavior"]["KEEP"]["passed"]
    assert not report["passed"]
    assert report["by_behavior"]["FOLLOW"]["mean_failure_fraction"] == pytest.approx(
        0.2
    )
    assert report["by_behavior"]["FOLLOW"]["stochastic_pass_rate"] == 1.0
    for row in rows:
        row.update(
            mean_pass=True,
            samples=256,
            accepted=230 if row["behavior"] == "FOLLOW" else 256,
        )
    assert not summarize_acceptance(rows)[
        "passed"
    ]  # 230/256 < .9, even if pooled total exceeds .9
    for row in rows:
        row["accepted"] = 231
    assert summarize_acceptance(rows)["passed"]


def test_missing_or_smoke_evidence_never_enables_sac():
    with pytest.raises(ValueError):
        require_training_gate({}, {"v": 1}, "abc")
    report = dict(versions={"v": 1}, checkpoint_sha256="abc", scope="SMOKE")
    with pytest.raises(ValueError, match="smoke"):
        require_training_gate(report, {"v": 1}, "abc")
    report.update(
        scope="QUALIFIED_BC_GATE",
        states=[
            dict(behavior=b, mean_pass=True, samples=256, accepted=256)
            for b in ("KEEP", "FOLLOW")
        ],
    )
    with pytest.raises(ValueError, match="synthetic"):
        require_training_gate(report, {"v": 1}, "abc")
