"""Approved FOLLOW semantics: soft desired gap, hard collision/no passing."""

from dataclasses import replace
import numpy as np
import pytest

from tests.environment.test_womd_control import scene
from src.contracts.planning import Source
from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    scripted_proposal,
)
from src.environment.womd_control.control import Trajectory, _following_order
from src.training.sac.collector import collect_step


def following_scene(offset=10.0, lead_speed=10.0, lateral=False):
    proto = scene()
    for i, state in enumerate(proto.tracks[1].states):
        state.center_x = offset + lead_speed * i * 0.1
        state.velocity_x = lead_speed
    if lateral:
        for track, y in zip(proto.tracks, (-0.6, 0.6)):
            for state in track.states:
                state.center_y = y
                state.width = 0.6
    return proto


def test_safe_close_following_executes_and_keeps_gap_reward():
    e = SACPlanningEnvironment()
    b = e.reset(following_scene(), provider_spec="fixed_follow")
    target_id = e.command.draft.target_agent_id
    collected = collect_step(e, b, scripted_proposal)
    assert collected.result.check_result.valid
    assert collected.result.source == Source.NEW
    # At 10 m/s, desired gap=18 m but actual bumper gap=6 m.
    assert collected.reward.terms["gap"] == pytest.approx(-0.5)
    assert collected.reward.terms["raw_reject"] == 0.0
    assert collected.reward.terms["command_violation"] == 0.0
    assert e.command.draft.target_agent_id == target_id


def test_collision_is_still_rejected_below_desired_gap():
    e = SACPlanningEnvironment()
    e.reset(following_scene(offset=15.0, lead_speed=0.0), provider_spec="fixed_follow")
    check = e._check(Trajectory(scripted_proposal(e, e.token)), e._check_objects())
    assert not check.valid and check.reason == "PREDICTED_COLLISION"


def test_laterally_separated_passing_is_rejected_even_without_collision():
    e = SACPlanningEnvironment()
    e.reset(
        following_scene(offset=10.0, lead_speed=5.0, lateral=True),
        provider_spec="fixed_follow",
    )
    trajectory = Trajectory(scripted_proposal(e, e.token))
    from src.environment.womd_control.control import check_trajectory

    unconstrained = check_trajectory(
        trajectory, e.ego, e.route, e._check_objects(), e.memory, e.tick
    )
    assert unconstrained.valid, unconstrained
    check = e._check(trajectory, e._check_objects())
    assert check.reason == "COMMAND_VIOLATION" and check.violations == (
        "follow_overtaking",
    )


def test_bumper_order_uses_geometry_and_target_motion():
    e = SACPlanningEnvironment()
    e.reset(following_scene(), provider_spec="fixed_follow")
    ego = e.ego
    target = e.target_cache
    poses = [ego, replace(ego, position=ego.position + np.array([7.0, 0.0]))]
    np.testing.assert_array_equal(
        _following_order(poses, e.route, target, np.zeros(2)), [True, False]
    )
    np.testing.assert_array_equal(
        _following_order(poses, e.route, target, np.array([0.0, 1.0])), [True, True]
    )


def test_virtual_pd_overtaking_is_rejected(monkeypatch):
    from src.contracts.planning import ControlCommand
    import src.environment.womd_control.control as control

    e = SACPlanningEnvironment()
    e.reset(following_scene(offset=4.4, lateral=True), provider_spec="fixed_follow")
    trajectory = Trajectory(scripted_proposal(e, e.token))

    def overshoot(trajectory, ego, memory, tick, source=Source.NEW):
        return ControlCommand(tick, 1.0, 0.0, source, False, None), replace(
            memory, last_a=1.0, last_kappa=0.0
        )

    monkeypatch.setattr(control, "pd_control", overshoot)
    check = e._check(trajectory, e._check_objects())
    assert check.reason == "COMMAND_VIOLATION" and check.violations == (
        "virtual_follow_overtaking",
    )


def test_actual_unpredicted_target_motion_records_command_violation():
    e = SACPlanningEnvironment()
    b = e.reset(following_scene(lateral=True), provider_spec="fixed_follow")
    log = e._state.log_trajectory
    # Exogenous future logs are invisible to prediction; measure after stepping.
    e._state = e._state.replace(
        log_trajectory=log.replace(
            x=log.x.at[1, e.frame + 1].set(np.float32(e.ego.position[0] - 6.0))
        )
    )
    collected = collect_step(e, b, scripted_proposal)
    assert collected.result.check_result.valid
    assert "FOLLOW_OVERTAKING" in collected.result.event_flags
    assert collected.reward.terms["command_violation"] == -0.5
    assert collected.reward.terms["raw_reject"] == 0.0
