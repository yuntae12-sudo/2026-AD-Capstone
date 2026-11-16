from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from src.contracts.observation import CoreObservation
from src.contracts.planning import Maneuver, Source, TerminalReason, decode_waypoints
from src.environment.womd_control.control import Ego
from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    scripted_proposal,
)
from src.training.sac.collector import PendingTransition, collect_step
from src.training.sac.replay import Replay, Transition
from src.training.sac.reward import calculate_reward, collision_ttc
from src.training.sac.teacher import FrenetTeacher
from tests.environment.test_womd_control import scene


def test_ttc_uses_actual_oriented_boxes_and_constant_velocity():
    ego = Ego(np.zeros(2), np.array([5.0, 0.0]), np.zeros(2), True, 0.0, 4.0, 2.0)
    target = replace(ego, position=np.array([14.0, 0.0]), velocity=np.zeros(2))
    assert collision_ttc(ego, [target]) == pytest.approx(2.0)
    assert np.isinf(
        collision_ttc(ego, [replace(target, position=np.array([14.0, 3.0]))])
    )
    assert collision_ttc(ego, [replace(target, position=np.array([4.0, 0.0]))]) == 0.0
    assert collision_ttc(ego, [replace(target, yaw=np.pi / 2)]) == pytest.approx(2.2)


def reward_args():
    return dict(
        delta_s=1.0,
        speed=10.0,
        speed_limit=15.0,
        gap=18.0,
        ttc=float("inf"),
        acceleration=0.0,
        jerk=0.0,
        lateral_acceleration=0.0,
        tracking_error=0.0,
        raw_rejected=False,
        source=Source.NEW,
        terminal_reason=None,
    )


def test_reward_snapshot_failure_precedence_and_follow_stopping():
    draft = SimpleNamespace(
        maneuver=Maneuver.FOLLOW,
        desired_speed_mps=15.0,
        minimum_gap_m=3.0,
        time_gap_s=1.5,
    )
    args = reward_args()
    args.update(speed=0.0, gap=3.0, delta_s=0.0)
    reward = calculate_reward(draft, **args)
    assert reward.training == 0.0 and "speed" not in reward.terms
    args.update(
        terminal_reason=TerminalReason.ROUTE_GOAL, flags=("COLLISION", "OFFROAD")
    )
    assert calculate_reward(draft, **args).event == -2000.0
    args.update(terminal_reason=TerminalReason.DATA_CORRUPTION)
    reward = calculate_reward(draft, **args)
    assert not reward.sample_valid and reward.training == 0.0
    args = reward_args()
    args.update(raw_rejected=True, source=Source.REUSED)
    reward = calculate_reward(draft, **args)
    assert (
        reward.terms["raw_reject"] == -0.5 and reward.terms["failure_fallback"] == -0.5
    )


def test_teacher_roundtrip_same_actual_boundary_and_rejected_raw_replay():
    e = SACPlanningEnvironment()
    b = e.reset(scene())
    teacher = FrenetTeacher()
    proposed = teacher.propose(e, b.token).proposal
    np.testing.assert_allclose(
        decode_waypoints(
            proposed.raw_u,
            proposed.anchor_world_xy,
            proposed.anchor_yaw,
            proposed.anchor_velocity_world,
            proposed.anchor_acceleration_world,
        ),
        proposed.xy_world,
        atol=1e-10,
    )
    replay = Replay(4)
    first = collect_step(e, b, teacher.propose, replay, teacher=True)
    assert first.transition.teacher_accepted
    np.testing.assert_array_equal(
        first.transition.next_observation.flatten(),
        first.boundary.sac_observation.flatten(),
    )
    b = first.boundary
    # Generate consistent raw u for a finite physically invalid proposal.
    invalid = e.proposal_from_world(
        scripted_proposal(e, b.token).xy_world + np.array([0.0, 20.0]), b.token
    )
    second = collect_step(e, b, lambda *_: invalid, replay)
    assert not second.result.check_result.valid
    assert replay.size == 2
    np.testing.assert_array_equal(replay.arrays["raw_u"][1], invalid.raw_u)
    np.testing.assert_array_equal(
        replay.arrays["next_obs"][1], second.boundary.sac_observation.flatten()
    )


def test_replay_bounded_storage_and_exact_rng_resume(tmp_path):
    obs = CoreObservation.empty()
    t = Transition(
        obs,
        np.ones((30, 2), np.float64),
        0.2,
        obs,
        1,
        True,
        True,
        False,
        "episode",
        "scene",
        0,
    )
    replay = Replay(3, seed=5, versions={"schema": 3})
    for i in range(5):
        replay.add(replace(t, reward=float(i), tick=i))
    assert replay.size == 3 and replay.cursor == 2
    assert replay.nbytes / 3 * 50000 < 4 * 1024**3
    with pytest.raises(MemoryError):
        Replay(50000, ram_limit=1024)
    assert not replay.add(replace(t, sample_valid=False))
    with pytest.raises(ValueError):
        replay.add(replace(t, raw_u=np.full((30, 2), np.nan)))
    replay.save(tmp_path / "replay.npz")
    restored = Replay.load(tmp_path / "replay.npz", {"schema": 3})
    np.testing.assert_array_equal(
        replay.sample(8)["reward"], restored.sample(8)["reward"]
    )
    with pytest.raises(ValueError):
        Replay.load(tmp_path / "replay.npz", {"schema": 2})


def test_pending_forced_ticks_discount_and_final_observation():
    e = SACPlanningEnvironment()
    b = e.reset(scene())
    proposal = scripted_proposal(e, b.token)
    pending = PendingTransition(b.sac_observation, proposal)
    reward = SimpleNamespace(training=1.0, sample_valid=True)
    result = SimpleNamespace(
        pre_token=b.token,
        post_tick=1,
        sample_valid=True,
        terminated=False,
        truncated=False,
    )
    assert pending.append(reward, result, replace(b, needs_sac_action=False)) is None
    result = SimpleNamespace(
        pre_token=replace(b.token, sim_tick=1),
        post_tick=2,
        sample_valid=True,
        terminated=False,
        truncated=True,
    )
    transition = pending.append(reward, result, replace(b, needs_sac_action=False))
    assert transition.reward == pytest.approx(1.99)
    assert transition.delta_ticks == 2 and not transition.bootstrap
    assert transition.next_observation is b.sac_observation
    with pytest.raises(ValueError):
        pending.append(reward, result, b)
    invalid = PendingTransition(b.sac_observation, proposal)
    result.pre_token = b.token
    result.post_tick = 1
    result.sample_valid = False
    assert invalid.append(reward, result, b) is None and invalid.closed


def test_teacher_uses_independent_velocity_and_vector_acceleration():
    e = SACPlanningEnvironment()
    b = e.reset(scene())
    teacher = FrenetTeacher()
    teacher.propose(e, b.token)
    ego = replace(
        e.ego,
        velocity=np.array([5.0, 1.0]),
        acceleration=np.array([2.0, -0.4]),
        yaw=0.7,
    )
    f = teacher.anchor(ego)
    r, v, a, n, dn, ddn = teacher.frame(f.s)
    np.testing.assert_allclose(r + f.d * n, ego.position, atol=1e-6)
    np.testing.assert_allclose(
        (v + f.d * dn) * f.s_d + n * f.d_d, ego.velocity, atol=1e-12
    )
    reconstructed = (
        (v + f.d * dn) * f.s_dd
        + n * f.d_dd
        + (a + f.d * ddn) * f.s_d**2
        + 2 * dn * f.s_d * f.d_d
    )
    np.testing.assert_allclose(reconstructed, ego.acceleration, atol=1e-12)


def test_raw_policy_action_is_not_reencoded_after_world_rounding():
    from src.training.sac.policy import proposal_from_action

    e = SACPlanningEnvironment()
    b = e.reset(scene())
    raw = np.random.default_rng(42).normal(0, 0.001, (30, 2)).astype(np.float64)
    proposal = proposal_from_action(e, b.token, raw)
    np.testing.assert_array_equal(proposal.raw_u, raw)
    teacher = FrenetTeacher()
    first = teacher.propose(e, b.token).proposal
    log = e._state.log_trajectory
    e._state = e._state.replace(
        log_trajectory=log.replace(x=log.x.at[:, e.frame + 1 :].set(9999))
    )
    np.testing.assert_array_equal(
        teacher.propose(e, b.token).proposal.xy_world, first.xy_world
    )


def test_empty_replay_checkpoint_streaming(tmp_path):
    replay = Replay(2, versions={"test": 1})
    replay.save(tmp_path / "empty.npz")
    restored = Replay.load(tmp_path / "empty.npz", {"test": 1})
    assert restored.size == 0 and restored.cursor == 0
