from dataclasses import replace
import numpy as np
from src.contracts.planning import CommandRequest, ExecutionRequest
from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    scripted_proposal,
)
from src.environment.womd_control.control import Trajectory, pd_control, PDMemory
from tests.environment.test_womd_control import scene


def test_semantic_command_change_clears_errors_but_retains_actual_inputs():
    e = SACPlanningEnvironment()
    e.reset(scene())
    e.step_waypoints(scripted_proposal(e, e.token), e.token)
    previous = e.memory
    request = CommandRequest(
        "change-speed",
        None,
        replace(e.command.draft, desired_speed_mps=8.0),
        ExecutionRequest.EXECUTE,
        "fixed",
    )
    receipt, b = e.submit_behavior(request, e.token)
    assert receipt.accepted and e.command.command_id == 2
    assert not e.memory.errors_valid and e.memory.applied_control_valid
    assert (
        e.memory.last_a == previous.last_a
        and e.memory.last_kappa == previous.last_kappa
    )
    trajectory = Trajectory(scripted_proposal(e, b.token))
    # Inactive old derivatives cannot contaminate the first new-command input.
    dirty = replace(
        e.memory,
        previous_e_y=100.0,
        previous_e_v=-100.0,
        filtered_de_y=999.0,
        filtered_de_v=-999.0,
    )
    a, am = pd_control(trajectory, e.ego, e.memory, e.tick)
    z, zm = pd_control(trajectory, e.ego, dirty, e.tick)
    assert a == z and am == zm


def test_same_command_path_replacement_preserves_memory_and_check_is_private():
    e = SACPlanningEnvironment()
    e.reset(scene())
    e.step_waypoints(scripted_proposal(e, e.token), e.token)
    memory = e.memory
    proposal = scripted_proposal(e, e.token)
    trajectory = Trajectory(proposal)
    assert e._check(trajectory, e._check_objects()).valid
    assert e.memory == memory and e.memory.errors_valid
    _, expected = pd_control(trajectory, e.ego, memory, e.tick)
    e.step_waypoints(proposal, e.token)
    assert e.memory.previous_e_v == expected.previous_e_v
    assert e.memory.filtered_de_v == expected.filtered_de_v


def test_endpoint_only_stop_reversal_is_not_a_valid_reference():
    e = SACPlanningEnvironment()
    e.reset(scene())
    t = np.arange(1, 31) * 0.1
    s = t / 3.0
    distance = 3 * e.ego.speed * (s - s**3 + 0.5 * s**4)
    xy = e.ego.position + np.column_stack([distance, np.zeros(30)])
    trajectory = Trajectory(e.proposal_from_world(xy, e.token))
    assert trajectory.evaluate(3.0, 1)[0] < -0.01
    assert not e._check(trajectory, e._check_objects()).valid


def test_near_stationary_drift_does_not_replace_measured_body_heading():
    proto = scene()
    for state in proto.tracks[0].states:
        state.center_x = 2.0
        state.velocity_x = 0.0003
        state.velocity_y = 0.0002
        state.heading = 0.0
    e = SACPlanningEnvironment()
    e.reset(proto)
    proposal = e.proposal_from_world(
        np.repeat(e.ego.position[None], 30, axis=0), e.token
    )
    trajectory = Trajectory(proposal)
    # Initial drift is retained, not clipped to zero or aligned to yaw.
    np.testing.assert_array_equal(trajectory.evaluate(0.0, 1), e.ego.velocity)
    np.testing.assert_array_equal(
        trajectory.yaws(np.array([0.0, 0.02, 0.1, 1.0, 3.0])), np.zeros(5)
    )
    check = e._check(trajectory, e._check_objects())
    assert check.valid, check
