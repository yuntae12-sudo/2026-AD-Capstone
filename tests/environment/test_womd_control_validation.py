"""Ensure failed/partial rollouts cannot masquerade as full successful tests."""

import pytest

from src.environment.womd_control.validation import summarize, tracking_summary


def row(source="NEW", tracking=None):
    return dict(
        source=source,
        tracking_error=tracking,
        check={"reason": "OK"},
        attempts=[],
        speed_mps=5.0,
        wall_time_s=0.2,
        route_progress_m=1.0,
        terminated=False,
        truncated=False,
        terminal_reason=None,
    )


def test_emergency_without_reference_is_not_zero_error():
    measured = dict(position_m=0.4, speed_mps=0.2, heading_deg=None)
    result = tracking_summary([row(tracking=measured), row("EMERGENCY")])
    assert result["samples"] == 1 and result["without_reference"] == 1
    assert result["position_m"]["rms"] == pytest.approx(0.4)
    assert result["heading_deg"]["rms"] is None


@pytest.mark.parametrize(
    "reason,expected",
    [
        (None, "STEP_LIMIT_REACHED"),
        ("COLLISION", "FAILURE"),
        ("ROUTE_GOAL", "ROUTE_GOAL"),
        ("RECORD_END", "RECORD_END"),
    ],
)
def test_rollout_outcomes_are_explicit(reason, expected):
    step = row()
    step.update(
        terminal_reason=reason,
        terminated=reason in ("COLLISION", "ROUTE_GOAL"),
        truncated=reason == "RECORD_END",
    )
    report = summarize(dict(status="RUNNING", steps=[step]))
    assert report["outcome"] == expected
    assert report["episode_ended"] == (reason is not None)
    assert report["simulated_duration_s"] == 0.1


def test_runtime_error_keeps_partial_progress_without_claiming_completion():
    report = summarize(
        dict(status="RUNTIME_ERROR", steps=[row()], reason="unexpected error")
    )
    assert report["outcome"] == "RUNTIME_ERROR" and not report["episode_ended"]
    assert report["steps_executed"] == 1


def test_candidate_rejections_are_counted_separately_from_selected_source():
    step = row("REUSED", dict(position_m=0.001, speed_mps=0.002, heading_deg=0.1))
    step["check"]["reason"] = "PREDICTED_OFFROAD"
    step["attempts"] = [
        dict(
            source="NEW",
            check=dict(valid=False, reason="PREDICTED_OFFROAD", violations=[]),
        ),
        dict(source="REUSED", check=dict(valid=True, reason="OK", violations=[])),
    ]
    report = summarize(dict(status="RUNNING", steps=[step]))
    assert report["execution_sources"] == {"REUSED": 1}
    assert report["rejected_candidates"] == {"NEW:PREDICTED_OFFROAD": 1}
    assert report["tracking"]["samples"] == 1
