"""Diagnostic instrumentation must preserve production decisions and state."""

from dataclasses import replace
import numpy as np
import pytest

from scripts.diagnose_sac_replanning import explain_check
from src.environment.womd_control import control
from src.environment.womd_control.environment import SACPlanningEnvironment
from src.training.sac.policy import proposal_from_action
from tests.environment.test_womd_control import scene


@pytest.mark.parametrize("kind", ["accepted", "offroad", "rough"])
def test_observed_checker_preserves_decision_state_and_functions(kind):
    e = SACPlanningEnvironment()
    e.reset(scene())
    action = np.zeros((30, 2))
    if kind == "offroad":
        # Cubic lateral drift starts with the measured zero acceleration.
        action[:, 1] = np.arange(1, 31) * 0.1 / 30
    elif kind == "rough":
        action[:, 1] = np.where(np.arange(30) % 2, 0.01, -0.01)
    trajectory = control.Trajectory(proposal_from_action(e, e.token, action))
    expected = e._check(trajectory, e._check_objects())
    state, token, memory, plan = e._state, e.token, e.memory, e.old_plan
    original_scene, original_pd = control._scene_checks, control.pd_control
    result = explain_check(e, trajectory)
    assert result["check"]["valid"] == expected.valid
    assert result["check"]["reason"] == expected.reason
    assert e._check(trajectory, e._check_objects()) == expected
    assert e._state is state and e.token == token and e.memory == memory
    assert e.old_plan is plan
    assert control._scene_checks is original_scene and control.pd_control is original_pd
    if kind == "accepted":
        assert result["decision_phase"] == "accepted" and expected.valid
    else:
        assert result["decision_phase"] == "reference" and not expected.valid
    if kind == "offroad":
        assert expected.reason == "PREDICTED_OFFROAD"
        assert result["scene_events"][0]["phase"] == "reference"


def test_virtual_pd_failure_is_distinguished_from_reference_failure():
    e = SACPlanningEnvironment()
    e.reset(scene())
    # The same straight reference is valid, but a large previous curvature
    # prevents PD from returning to zero within its per-tick slew limit.
    trajectory = control.Trajectory(proposal_from_action(e, e.token, np.zeros((30, 2))))
    assert e._check(trajectory, e._check_objects()).valid
    e.memory = replace(e.memory, last_kappa=0.3, applied_control_valid=True)
    expected = e._check(trajectory, e._check_objects())
    result = explain_check(e, trajectory)
    assert result["decision_phase"] == "virtual_pd"
    assert result["check"]["violations"] == ("virtual_lateral_acceleration",)
    assert result["check"]["reason"] == expected.reason
    assert e._check(trajectory, e._check_objects()) == expected
