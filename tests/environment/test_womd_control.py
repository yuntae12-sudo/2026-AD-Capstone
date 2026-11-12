"""Behavioral regressions for the new single-scene control boundary."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import replace
import numpy as np
import pytest

from src.contracts.planning import (
    CommandRequest,
    ContractError,
    ExecutionRequest,
    Maneuver,
    Source,
    TerminalReason,
    decode_waypoints,
)
from src.environment.womd_control.control import (
    Ego,
    PDMemory,
    Trajectory,
    advance,
    pd_control,
)
from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    scripted_proposal,
)
from src.environment.womd_control.geometry import (
    ResetRejected,
    box,
    contains_box,
    overlaps,
)
from src.scenarios.vendor.waymo_open_dataset.protos import scenario_pb2


def scene():
    proto = scenario_pb2.Scenario(
        scenario_id="straight-control", sdc_track_index=0, current_time_index=0
    )
    proto.timestamps_seconds.extend(np.arange(200) * 0.1)
    for object_id, offset in ((11, 0), (22, 80)):
        track = proto.tracks.add(id=object_id, object_type=1)
        for t in range(200):
            track.states.add(
                center_x=offset + t,
                center_y=0,
                center_z=0,
                velocity_x=10,
                velocity_y=0,
                heading=0,
                length=4,
                width=2,
                height=1.5,
                valid=True,
            )
    lane = proto.map_features.add(id=1).lane
    lane.type = 1
    lane.speed_limit_mph = 30
    xs = np.arange(-20, 251, 5)
    for x in xs:
        lane.polyline.add(x=x, y=0, z=0)
    for feature_id, y, kind, reverse in (
        (2, 2, "road_line", False),
        (3, -2, "road_line", False),
        (4, 6, "road_edge", True),
        (5, -6, "road_edge", False),
    ):
        feature = getattr(proto.map_features.add(id=feature_id), kind)
        feature.type = 1
        for x in xs[::-1] if reverse else xs:
            feature.polyline.add(x=x, y=y, z=0)
    lane.left_boundaries.add(
        lane_start_index=0,
        lane_end_index=len(xs) - 1,
        boundary_feature_id=2,
        boundary_type=1,
    )
    lane.right_boundaries.add(
        lane_start_index=0,
        lane_end_index=len(xs) - 1,
        boundary_feature_id=3,
        boundary_type=1,
    )
    return proto


@pytest.fixture
def environment():
    e = SACPlanningEnvironment()
    e.reset(scene())
    return e


def test_reset_current_history_observation_and_one_controlled_tick(environment):
    e = environment
    token = e.token
    boundary = e._boundary()
    assert token.sim_tick == 0 and e.frame == 2
    assert boundary.sac_observation.flatten().shape == (5975,)
    proposal = scripted_proposal(e, token)
    before = e.ego
    result, boundary = e.step_waypoints(proposal, token)
    assert result.source == Source.NEW and result.check_result.valid
    assert e.tick == 1 and e.frame == 3 and int(e._state.timestep) == 3
    assert boundary.token != token and boundary.needs_sac_action
    expected = advance(
        before,
        result.actual_control.acceleration_mps2,
        result.actual_control.curvature_per_m,
    )
    np.testing.assert_allclose(e.ego.position, expected.position, atol=1e-5)
    with pytest.raises(ContractError, match="STALE"):
        e.step_waypoints(proposal, token)
    assert e.tick == 1


def test_ego_is_controlled_while_background_replays(environment):
    e = environment
    token = e.token
    # Change only future logs. Neither current policy features nor ego action can see them.
    initial = e._boundary().sac_observation.flatten().copy()
    log = e._state.log_trajectory
    log = log.replace(x=log.x.at[0, e.frame + 1 :].set(999).at[1, e.frame + 1].set(120))
    e._state = e._state.replace(log_trajectory=log)
    np.testing.assert_array_equal(e._boundary().sac_observation.flatten(), initial)
    e.step_waypoints(scripted_proposal(e, token), token)
    assert e.ego.position[0] < 10
    assert float(e._state.sim_trajectory.x[1, e.frame]) == 120


def test_new_background_object_retains_source_dimensions():
    proto = scene()
    proto.tracks[1].states[2].valid = False
    e = SACPlanningEnvironment()
    e.reset(proto)
    e.step_waypoints(scripted_proposal(e, e.token), e.token)
    assert bool(e._state.sim_trajectory.valid[1, 3])
    assert float(e._state.sim_trajectory.width[1, 3]) == 2
    assert float(e._state.sim_trajectory.length[1, 3]) == 4


def test_behavior_revision_heartbeat_idempotency_and_rejection(environment):
    e = environment
    before = e.token
    request = CommandRequest(
        "one", None, e.command.draft, ExecutionRequest.EXECUTE, "fixed"
    )
    receipt, boundary = e.submit_behavior(request, before)
    assert receipt.accepted and e.tick == 0 and e.counter == 1
    assert boundary.token.revision == 1
    duplicate, _ = e.submit_behavior(request, before)
    assert duplicate == receipt and e.revision == 1
    with pytest.raises(ContractError, match="DUPLICATE"):
        e.submit_behavior(replace(request, provider_version="different"), e.token)
    with pytest.raises(ContractError, match="STALE"):
        e.submit_behavior(replace(request, request_id="two"), before)
    changed = replace(
        request,
        request_id="new-speed",
        command_draft=replace(request.command_draft, desired_speed_mps=8.0),
    )
    receipt, _ = e.submit_behavior(changed, e.token)
    assert receipt.accepted and e.command.command_id == 2 and e.tick == 0
    unsupported = replace(
        changed,
        request_id="merge",
        command_draft=replace(
            changed.command_draft, maneuver=Maneuver.MERGE, target_lane_id=1
        ),
    )
    receipt, _ = e.submit_behavior(unsupported, e.token)
    assert not receipt.accepted and e.command.command_id == 2


def test_anchor_error_does_not_advance_time(environment):
    e = environment
    proposal = scripted_proposal(e, e.token)
    position = proposal.anchor_world_xy + np.array([0.1, 0])
    changed = replace(
        proposal,
        anchor_world_xy=position,
        xy_world=decode_waypoints(
            proposal.raw_u,
            position,
            proposal.anchor_yaw,
            proposal.anchor_velocity_world,
            proposal.anchor_acceleration_world,
        ),
    )
    with pytest.raises(ContractError, match="ANCHOR"):
        e.step_waypoints(changed, e.token)
    assert e.tick == 0


def test_invalid_finite_path_brakes_and_consumes_one_tick(environment):
    e = environment
    ego = e.ego
    xy = decode_waypoints(
        np.ones((30, 2)) * 10, ego.position, ego.yaw, ego.velocity, ego.acceleration
    )
    result, _ = e.step_waypoints(e.proposal_from_world(xy, e.token), e.token)
    assert not result.check_result.valid
    assert (
        result.source == Source.EMERGENCY
        and result.actual_control.acceleration_mps2 < 0
    )
    assert result.raw_proposal is not None and e.tick == 1


def test_reuse_preserves_generation_tick_and_expires_after_two_ticks(environment):
    e = environment
    e.step_waypoints(scripted_proposal(e, e.token), e.token)
    original_tick = e.old_plan.proposal.generated_tick
    sources = []
    for _ in range(3):
        ego = e.ego
        xy = decode_waypoints(
            np.ones((30, 2)) * 10, ego.position, ego.yaw, ego.velocity, ego.acceleration
        )
        result, _ = e.step_waypoints(e.proposal_from_world(xy, e.token), e.token)
        sources.append(result.source)
        assert e.old_plan.proposal.generated_tick == original_tick
    assert sources[:2] == [Source.REUSED, Source.REUSED]
    assert sources[2] in (Source.BRAKE, Source.EMERGENCY)
    assert e.feedback.reconsider_event


def test_heartbeat_keeps_command_and_pd_memory(environment):
    e = environment
    original_id = e.command.command_id
    for _ in range(5):
        e.step_waypoints(scripted_proposal(e, e.token), e.token)
    assert e.command.command_id == original_id
    assert e.command.created_tick == 0 and e.command.valid_until_tick == 15
    assert e.memory.errors_valid and e.memory.applied_control_valid


def test_record_end_terminates_once_without_advancing_past_log():
    proto = scene()
    proto.map_features[0].lane.speed_limit_mph = 6 / 0.44704
    del proto.timestamps_seconds[54:]
    for track in proto.tracks:
        del track.states[54:]
        for t, state in enumerate(track.states):
            state.velocity_x = 5
            state.center_x = (0 if track.id == 11 else 80) + 0.5 * t
    e = SACPlanningEnvironment()
    boundary = e.reset(proto)
    while not (boundary.terminated or boundary.truncated):
        _, boundary = e.step_waypoints(
            scripted_proposal(e, boundary.token), boundary.token
        )
    assert boundary.truncated and boundary.terminal_reason == TerminalReason.RECORD_END
    assert e.frame == 53 and e.tick == 51 and not boundary.needs_sac_action
    with pytest.raises(ContractError, match="already ended"):
        e.step_forced(boundary.token)


@pytest.mark.parametrize("track_index", [0, 1])
def test_corrupt_new_object_is_diagnostic_not_a_training_transition(track_index):
    proto = scene()
    proto.tracks[track_index].states[3].width = 0
    e = SACPlanningEnvironment()
    e.reset(proto)
    result, boundary = e.step_waypoints(scripted_proposal(e, e.token), e.token)
    assert (
        boundary.terminated
        and boundary.terminal_reason == TerminalReason.DATA_CORRUPTION
    )
    assert not result.sample_valid and not boundary.needs_sac_action


def test_partial_map_outside_current_covered_section_is_not_invented():
    proto = scene()
    lane = proto.map_features[0].lane
    lane.left_boundaries[0].lane_start_index = 1
    lane.right_boundaries[0].lane_start_index = 1
    e = SACPlanningEnvironment()
    e.reset(proto)
    assert e.route.center[0, 0] == -15
    assert not e.route.contains(np.array([-18.0, 0]), 0, 4, 2)


@pytest.mark.parametrize("fault", ["boundary", "speed", "jitter", "short", "signal"])
def test_reset_excludes_unsupported_data(fault):
    proto = scene()
    if fault == "boundary":
        del proto.map_features[0].lane.left_boundaries[:]
    elif fault == "speed":
        proto.map_features[0].lane.speed_limit_mph = 0
    elif fault == "jitter":
        proto.timestamps_seconds[4] += 0.02
    elif fault == "short":
        del proto.map_features[0].lane.polyline[10:]
    elif fault == "signal":
        for _ in range(200):
            proto.dynamic_map_states.add().lane_states.add(lane=1, state=6)
    with pytest.raises(ResetRejected):
        SACPlanningEnvironment().reset(proto)


def test_reset_episode_isolation(environment):
    e = environment
    old = e.token
    e.step_waypoints(scripted_proposal(e, old), old)
    boundary = e.reset(scene())
    assert boundary.token.episode_uid != old.episode_uid
    assert (
        e.tick == 0
        and e.counter == 1
        and not e.memory.applied_control_valid
        and e.old_plan is None
    )


def test_follow_target_grace_never_rebinds():
    e = SACPlanningEnvironment()
    e.reset(scene(), provider_spec="fixed_follow")
    assert e.command.draft.target_agent_id == 22
    log = e._state.log_trajectory
    e._state = e._state.replace(
        log_trajectory=log.replace(valid=log.valid.at[1, e.frame + 1 :].set(False))
    )
    for age in (1, 2, 3):
        result, boundary = e.step_waypoints(scripted_proposal(e, e.token), e.token)
        assert e.command.draft.target_agent_id == 22
        if age < 3:
            assert not boundary.terminated and e.target_estimated
        else:
            assert (
                boundary.terminated
                and boundary.terminal_reason == TerminalReason.TASK_LOST
            )


def test_pd_virtual_model_matches_waymax_equations_and_limits():
    ego = Ego(
        np.array([10.0, 20.0]), np.array([3.0, 2.0]), np.zeros(2), False, 0.4, 4.0, 2.0
    )
    result = advance(ego, 1.2, 0.1)
    expected_position = (
        ego.position
        + ego.velocity * 0.1
        + 0.5 * 1.2 * np.array([np.cos(0.4), np.sin(0.4)]) * 0.01
    )
    np.testing.assert_allclose(result.position, expected_position)
    assert result.speed == pytest.approx(ego.speed + 0.12)
    assert result.yaw == pytest.approx(0.4 + 0.1 * (ego.speed * 0.1 + 0.5 * 1.2 * 0.01))


def test_virtual_model_matches_installed_waymax_with_velocity_heading_mismatch():
    import jax
    import jax.numpy as jnp
    from waymax import datatypes, dynamics
    from src.scenarios.womd20 import scenario_to_waymax

    state, _ = scenario_to_waymax(scene())
    trajectory = jax.tree_util.tree_map(lambda x: x[:1, :1], state.log_trajectory)
    trajectory = trajectory.replace(
        x=jnp.array([[10.0]], jnp.float32),
        y=jnp.array([[20.0]], jnp.float32),
        vel_x=jnp.array([[3.0]], jnp.float32),
        vel_y=jnp.array([[2.0]], jnp.float32),
        yaw=jnp.array([[0.4]], jnp.float32),
    )
    update = dynamics.InvertibleBicycleModel(
        dt=0.1, normalize_actions=False
    ).compute_update(
        datatypes.Action(
            data=jnp.array([[1.2, 0.1]], jnp.float32), valid=jnp.array([[True]])
        ),
        trajectory,
    )
    ego = Ego(
        np.array([10.0, 20.0]), np.array([3.0, 2.0]), np.zeros(2), False, 0.4, 4.0, 2.0
    )
    expected = advance(ego, 1.2, 0.1)
    np.testing.assert_allclose(
        [float(update.x[0, 0]), float(update.y[0, 0])], expected.position, atol=2e-6
    )
    np.testing.assert_allclose(
        [float(update.vel_x[0, 0]), float(update.vel_y[0, 0])],
        expected.velocity,
        atol=2e-6,
    )
    assert float(update.yaw[0, 0]) == pytest.approx(expected.yaw, abs=2e-6)


def test_actual_collision_takes_priority_over_goal(environment):
    e = environment
    e.route = replace(e.route, goal_s=e.route.start_s + 0.5)
    log = e._state.log_trajectory
    e._state = e._state.replace(
        log_trajectory=log.replace(x=log.x.at[1, e.frame + 1].set(3))
    )
    result, boundary = e.step_waypoints(scripted_proposal(e, e.token), e.token)
    assert boundary.terminated and result.terminal_reason == TerminalReason.COLLISION
    assert not e.goal_satisfied


def test_quintic_anchor_and_c2_and_preview_memory(environment):
    e = environment
    trajectory = Trajectory(scripted_proposal(e, e.token))
    np.testing.assert_allclose(trajectory.evaluate(0), e.ego.position)
    np.testing.assert_allclose(trajectory.evaluate(0, 1), e.ego.velocity)
    np.testing.assert_allclose(trajectory.evaluate(0, 2), e.ego.acceleration)
    for knot in np.arange(0.1, 3, 0.1):
        np.testing.assert_allclose(
            trajectory.evaluate(knot - 1e-8, 1),
            trajectory.evaluate(knot + 1e-8, 1),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            trajectory.evaluate(knot - 1e-8, 2),
            trajectory.evaluate(knot + 1e-8, 2),
            atol=1e-5,
        )
    control, memory = pd_control(trajectory, e.ego, e.memory, 0)
    assert abs(control.acceleration_mps2) <= 2 and abs(control.curvature_per_m) <= 0.05
    assert memory.errors_valid and not e.memory.errors_valid
    with pytest.raises(ValueError, match="extrapolation"):
        trajectory.evaluate(3.1)


def test_corridor_checks_entire_rectangle_and_touching_collision():
    a = box(np.zeros(2), 0, 4, 2)
    b = box(np.array([4.0, 0.0]), 0, 4, 2)
    assert overlaps(a, b)
    assert not overlaps(a, b + np.array([0.01, 0]))
    # A notch cuts through a rectangle edge even when all its corners are inside.
    polygon = np.array(
        [[-3, -3], [3, -3], [3, 3], [0.5, 3], [0.5, 0], [-0.5, 0], [-0.5, 3], [-3, 3]],
        float,
    )
    assert not contains_box(polygon, a)
