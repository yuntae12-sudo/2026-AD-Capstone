"""Optimization oracle: original scalar predicates retained only in tests."""

import numpy as np
import pytest
from dataclasses import replace

from src.environment.womd_control.geometry import box, contains_box, inside, project
from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    scripted_proposal,
)
from src.contracts.planning import TerminalReason, ControlCommand, Source
from src.environment.womd_control.control import (
    Ego,
    SceneObjects,
    CheckResult,
    _scene_check,
)
from src.environment.womd_control.geometry import overlaps
from tests.environment.test_womd_control import scene


def scalar_inside(polygon, points):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    results = []
    for p in points:
        crossing_y = (a[:, 1] > p[1]) != (b[:, 1] > p[1])
        denom = b[:, 1] - a[:, 1]
        x = a[:, 0] + (p[1] - a[:, 1]) * np.divide(
            b[:, 0] - a[:, 0], denom, out=np.zeros_like(denom), where=denom != 0
        )
        ray = bool(np.count_nonzero(crossing_y & (p[0] < x)) % 2)
        edge = b - a
        f = np.clip(
            np.sum((p - a) * edge, axis=1)
            / np.maximum(np.sum(edge * edge, axis=1), 1e-12),
            0,
            1,
        )
        on_edge = np.min(np.linalg.norm(p - (a + f[:, None] * edge), axis=1)) <= 1e-8
        results.append(ray or on_edge)
    return np.array(results)


def scalar_contains(polygon, rectangle):
    if not scalar_inside(polygon, rectangle).all():
        return False

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    c, d = polygon, np.roll(polygon, -1, axis=0)
    for a, b in zip(rectangle, np.roll(rectangle, -1, axis=0)):
        ab, cd = b - a, d - c
        if (
            (cross(ab, c - a) * cross(ab, d - a) < -1e-12)
            & (cross(cd, a - c) * cross(cd, b - c) < -1e-12)
        ).any():
            return False
    return True


@pytest.mark.parametrize("offset", [[0, 0], [100000.0, -200000.0]])
def test_vectorized_predicates_match_scalar_for_concavity_and_boundaries(offset):
    polygon = (
        np.array(
            [
                [-5, -5],
                [5, -5],
                [5, 5],
                [0.5, 5],
                [0.5, 0],
                [-0.5, 0],
                [-0.5, 5],
                [-5, 5],
            ]
        )
        + offset
    )
    rng = np.random.default_rng(20261116)
    points = np.vstack(
        [
            rng.uniform(-6, 6, (1000, 2)) + offset,
            polygon,
            polygon + [1e-8, 0],
            polygon + [1.01e-8, 0],
        ]
    )
    np.testing.assert_array_equal(
        inside(polygon, points), scalar_inside(polygon, points)
    )
    for p in points[:250]:
        rectangle = box(
            p, rng.uniform(-np.pi, np.pi), rng.uniform(0.01, 6), rng.uniform(0.01, 3)
        )
        assert contains_box(polygon, rectangle) == scalar_contains(polygon, rectangle)


def test_cached_route_matches_reference_and_is_immutable():
    e = SACPlanningEnvironment()
    e.reset(scene())
    route = e.route
    rng = np.random.default_rng(16)
    for position in rng.uniform([-30, -10], [260, 10], (200, 2)):
        actual, expected = route.project(position), project(route.center, position)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        rectangle = box(position, 0.1, 4, 2)
        assert route.contains(position, 0.1, 4, 2) == scalar_contains(
            route.polygon, rectangle
        )
    with pytest.raises(ValueError):
        route.center[0, 0] = 100
    assert route.coverage_exited(box(np.array([251.0, 0.0]), 0, 4, 2))
    assert not route.coverage_exited(box(np.array([10.0, 3.0]), 0, 4, 2))
    positions = rng.uniform([-30, -10], [260, 10], (200, 2))
    batched = route.project_many(positions)
    for i, p in enumerate(positions):
        for actual, expected in zip(batched, project(route.center, p)):
            np.testing.assert_array_equal(actual[i], expected)
    rectangles = np.array([box(p, 0.1, 4, 2) for p in positions])
    np.testing.assert_array_equal(
        route.contains_many(rectangles),
        [scalar_contains(route.polygon, r) for r in rectangles],
    )


def test_nonfinite_post_step_ego_is_truncated_with_finite_empty_observation():
    e = SACPlanningEnvironment()
    e.reset(scene())
    step = e._step

    def corrupt(state, action):
        state = step(state, action)
        sim = state.sim_trajectory
        return state.replace(
            sim_trajectory=sim.replace(x=sim.x.at[e.ego_index, e.frame + 1].set(np.nan))
        )

    e._step = corrupt
    result, boundary = e.step_waypoints(scripted_proposal(e, e.token), e.token)
    assert result.terminal_reason == TerminalReason.DATA_CORRUPTION
    assert result.truncated and not result.terminated and not result.sample_valid
    assert np.isfinite(boundary.sac_observation.flatten()).all()
    assert not boundary.sac_observation.flatten().any()
    assert np.isfinite(e.feedback.progress)


@pytest.mark.parametrize(
    "position,reason",
    [
        ([251.0, 0.0], TerminalReason.MAP_COVERAGE_EXIT),
        ([10.0, 3.0], TerminalReason.CORRIDOR_VIOLATION),
    ],
)
def test_coverage_exit_and_lateral_exit_are_distinct(position, reason):
    e = SACPlanningEnvironment()
    e.reset(scene())
    sim = e._state.sim_trajectory
    e._state = e._state.replace(
        sim_trajectory=sim.replace(
            x=sim.x.at[0, e.frame].set(position[0]),
            y=sim.y.at[0, e.frame].set(position[1]),
        )
    )
    # Isolate the active corridor classification from the independent road-edge metric.
    e._offroad = lambda *args: np.array([False])
    assert e._actual_failure() == reason


def test_vectorized_scene_broad_phase_matches_original_collision_decisions():
    class OpenRoute:
        def contains(self, *args):
            return True

    rng = np.random.default_rng(42)
    ego = Ego(
        np.array([100000.0, -200000.0]), np.zeros(2), np.zeros(2), True, 0.3, 4.0, 2.0
    )
    objects = [
        replace(
            ego,
            position=ego.position + rng.uniform(-10, 10, 2),
            velocity=rng.uniform(-3, 3, 2),
            yaw=rng.uniform(-np.pi, np.pi),
        )
        for _ in range(100)
    ]
    for elapsed in np.linspace(0, 3, 31):
        expected = None
        ego_box = box(ego.position, ego.yaw, ego.length, ego.width, 0.3)
        for o in objects:
            position = o.position + o.velocity * elapsed
            radius = 0.5 * (
                np.hypot(ego.length + 0.6, ego.width + 0.6)
                + np.hypot(o.length, o.width)
            )
            if np.linalg.norm(position - ego.position) <= radius and overlaps(
                ego_box, box(position, o.yaw, o.length, o.width)
            ):
                expected = "PREDICTED_COLLISION"
                break
        assert (
            _scene_check(ego, OpenRoute(), SceneObjects(objects), elapsed) == expected
        )
    assert _scene_check(ego, OpenRoute(), SceneObjects([]), 0) is None
    for separation in (4.3 - 1e-8, 4.3, 4.3 + 1e-8):
        e = replace(ego, position=np.zeros(2), yaw=0.0)
        o = replace(e, position=np.array([separation, 0.0]))
        expected = (
            "PREDICTED_COLLISION"
            if overlaps(box(e.position, 0, 4, 2, 0.3), box(o.position, 0, 4, 2))
            else None
        )
        assert _scene_check(e, OpenRoute(), SceneObjects([o]), 0) == expected


def test_recorded_control_and_memory_equal_float32_waymax_input():
    e = SACPlanningEnvironment()
    e.reset(scene())
    step = e._step
    captured = []

    def record(state, action):
        captured.append(np.asarray(action.data))
        return step(state, action)

    e._step = record
    p = scripted_proposal(e, e.token)
    # Isolate the applied-input boundary with values that round in float32.
    control = ControlCommand(
        e.tick, 1.2345678901, 0.012345678901, Source.NEW, False, None
    )
    result, _ = e._advance(p, CheckResult(True, "ROUNDING_TEST"), control, e.memory)
    assert result.actual_control.acceleration_mps2 != control.acceleration_mps2
    assert result.actual_control.curvature_per_m != control.curvature_per_m
    np.testing.assert_array_equal(
        captured[-1],
        [
            result.actual_control.acceleration_mps2,
            result.actual_control.curvature_per_m,
        ],
    )
    assert e.memory.last_a == result.actual_control.acceleration_mps2
    assert e.memory.last_kappa == result.actual_control.curvature_per_m
