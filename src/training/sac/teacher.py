"""Current-state adapter for the existing Frenet planner.

The old planner's feasibility diagnostics are preserved. Its sampled XY, not
its velocities/accelerations, goes through the common production boundary.
Only that boundary decides BC eligibility. This is not the original ASMC.
"""

from dataclasses import dataclass, replace
import numpy as np

from src.contracts.planning import ContractError, Maneuver
from src.environment.behavior_action import BehaviorAction, BehaviorObjective
from src.environment.womd_control.geometry import arc, project
from src.planning.frenet_planner import (
    EgoKinematicState,
    FollowInputs,
    PlanRequest,
    PlannerConfig,
    plan,
)
from src.planning.candidate_evaluator import (
    CollisionLimits,
    CurrentAgentState,
    FeasibilityLimits,
)
from src.planning.reference import ReferenceLine
from src.scenarios.lane_geometry import LanePolyline


@dataclass(frozen=True)
class TeacherOutput:
    proposal: object
    diagnostics: dict


class LegacyFrenetTeacher:
    version = "frenet-current-state-xy-v1"

    def __init__(self):
        self._route = None

    def propose(self, environment, token):
        e = environment
        ego = e.planning_anchor(token)
        command = e.command.draft
        if command.maneuver not in (Maneuver.KEEP, Maneuver.FOLLOW):
            raise ContractError("teacher supports gated KEEP/FOLLOW only")
        if self._route is not e.route:
            xy = e.route.center
            tangent = np.gradient(xy, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
            self._reference = ReferenceLine.from_lane_polyline(
                LanePolyline(e.route.lane_ids[0], 1, xy, tangent, arc(xy))
            )
            self._route = e.route
        desired = min(
            command.desired_speed_mps, e.route.speed_limit(ego.position), 15.0
        )
        follow = FollowInputs()
        if command.maneuver == Maneuver.FOLLOW:
            if (
                e.target_cache is None
                or e.target_last_tick is None
                or e.tick - e.target_last_tick > 2
            ):
                raise ContractError("FOLLOW target unavailable; never substitute KEEP")
            target = e.target_cache
            position = (
                target.position + target.velocity * (e.tick - e.target_last_tick) * 0.1
            )
            gap = (
                project(e.route.center, position)[0]
                - project(e.route.center, ego.position)[0]
                - (target.length + ego.length) / 2
            )
            desired = max(
                0.0,
                min(
                    desired,
                    target.speed
                    + 0.5
                    * (gap - command.minimum_gap_m - command.time_gap_s * ego.speed),
                ),
            )
            follow = FollowInputs(gap, target.speed - ego.speed)
        objects = e._objects()
        agents = [
            CurrentAgentState(
                i, *o.position, *o.velocity, 0.5 * np.hypot(o.length, o.width)
            )
            for i, o in objects.items()
        ]
        config = PlannerConfig(
            0.1,
            3.0,
            2.0,
            FeasibilityLimits(6.0, 0.3, 20.0, 0.0),
            CollisionLimits(0.5 * np.hypot(ego.length, ego.width), 1.0, 0.3),
        )
        action = (
            BehaviorAction.KEEP
            if command.maneuver == Maneuver.KEEP
            else BehaviorAction.FOLLOW
        )
        result = plan(
            PlanRequest(
                action,
                BehaviorObjective(desired, "source", None),
                EgoKinematicState(*ego.position, ego.yaw, ego.speed),
                self._reference,
                self._reference,
                follow,
                agents,
            ),
            config,
        )
        trajectory = result.cartesian_trajectory
        if trajectory is None:
            raise ContractError(
                f"teacher produced no raw path: {result.status}: {result.diagnostics}"
            )
        times = np.arange(1, 31) * 0.1
        if trajectory.t[0] > 0.1 + 1e-9 or trajectory.t[-1] < 3.0 - 1e-9:
            raise ContractError("teacher horizon does not cover required samples")
        xy = np.column_stack(
            [
                np.interp(times, trajectory.t, trajectory.x),
                np.interp(times, trajectory.t, trajectory.y),
            ]
        )
        proposal = e.proposal_from_world(xy, token, model_version=self.version)
        return TeacherOutput(
            proposal, dict(legacy_status=result.status, **result.diagnostics)
        )


class FrenetTeacher:
    """Reuse existing Frenet polynomials with a full current-state anchor.

    The legacy scalar-speed/zero-acceleration projection cannot represent the
    contract's independent vx/vy and vector acceleration. Here their derivatives
    are inverted in the planner's current-map Frenet frame. A C2 interpolant of
    the observed centerline supplies a differentiable frame; the original road
    polygon and all acceptance checks remain authoritative. No output repair.
    """

    version = "frenet-vector-anchor-c2-map-low-speed-v4"

    def __init__(self):
        self._route = None

    def frame(self, s):
        r = self._spline(s)
        v, a, j = (self._spline(s, i) for i in (1, 2, 3))
        speed = np.linalg.norm(v, axis=-1, keepdims=True)
        if np.any(speed < 1e-6):
            raise ContractError("degenerate teacher map tangent")
        dv = np.sum(v * a, axis=-1, keepdims=True) / speed
        ddv = (np.sum(a * a + v * j, axis=-1, keepdims=True) - dv**2) / speed
        tangent = v / speed
        dt = a / speed - v * dv / speed**2
        ddt = (
            j / speed
            - 2 * a * dv / speed**2
            - v * ddv / speed**2
            + 2 * v * dv**2 / speed**3
        )
        rotate = lambda x: np.stack([-x[..., 1], x[..., 0]], axis=-1)
        return r, v, a, rotate(tangent), rotate(dt), rotate(ddt)

    def anchor(self, ego):
        from scipy.optimize import minimize_scalar

        initial = project(self._route.center, ego.position)[0]
        low, high = max(0.0, initial - 3.0), min(self._arc[-1], initial + 3.0)
        solution = minimize_scalar(
            lambda s: np.sum((self._spline(s) - ego.position) ** 2),
            bounds=(low, high),
            method="bounded",
            options={"xatol": 1e-12},
        )
        s = float(solution.x)
        r, v, a, n, dn, ddn = self.frame(s)
        d = float((ego.position - r) @ n)
        matrix = np.column_stack([v + d * dn, n])
        if abs(np.linalg.det(matrix)) < 1e-6:
            raise ContractError("singular teacher Frenet frame")
        sd, dd = np.linalg.solve(matrix, ego.velocity)
        sdd, ddd = np.linalg.solve(
            matrix, ego.acceleration - (a + d * ddn) * sd**2 - 2 * dn * sd * dd
        )
        from src.planning.frenet_types import FrenetState

        return FrenetState(s, float(sd), float(sdd), d, float(dd), float(ddd))

    def propose(self, environment, token):
        from scipy.interpolate import CubicSpline
        from src.planning.candidate_generator import (
            generate_keep_candidate,
            generate_follow_or_merge_candidate,
        )

        e = environment
        ego = e.planning_anchor(token)
        draft = e.command.draft
        if draft.maneuver not in (Maneuver.KEEP, Maneuver.FOLLOW):
            raise ContractError("teacher supports gated KEEP/FOLLOW only")
        if self._route is not e.route:
            self._route = e.route
            self._arc = arc(e.route.center)
            self._spline = CubicSpline(
                self._arc, e.route.center, axis=0, extrapolate=False
            )
        anchor = self.anchor(ego)
        desired = min(draft.desired_speed_mps, e.route.speed_limit(ego.position), 15.0)
        if draft.maneuver == Maneuver.FOLLOW:
            if (
                e.target_cache is None
                or e.target_last_tick is None
                or e.tick - e.target_last_tick > 2
            ):
                raise ContractError("FOLLOW target unavailable")
            target = e.target_cache
            p = target.position + target.velocity * (e.tick - e.target_last_tick) * 0.1
            gap = (
                project(e.route.center, p)[0]
                - project(e.route.center, ego.position)[0]
                - (target.length + ego.length) / 2
            )
            desired = max(
                0.0,
                min(
                    desired,
                    target.speed
                    + 0.5 * (gap - draft.minimum_gap_m - draft.time_gap_s * ego.speed),
                ),
            )
        generator = (
            generate_keep_candidate
            if draft.maneuver == Maneuver.KEEP
            else generate_follow_or_merge_candidate
        )
        # Centering a lateral offset at near-zero speed can demand unbounded
        # curvature. Keep the current within-lane offset while below 2 m/s.
        # A near-stationary teacher can hold; KEEP/FOLLOW command identity and
        # desired-speed/gap reward remain unchanged. This is not goal success.
        lateral_offset = anchor.d if ego.speed < 2.0 else 0.0
        stationary_hold = (
            ego.speed <= 0.001 and np.linalg.norm(ego.acceleration) <= 0.01
        )
        if lateral_offset:
            anchor = replace(anchor, d=0.0)
        if stationary_hold:
            desired = 0.0
        # Central knot derivatives amplify the original polynomial's jerk.
        # A conservative *teacher generation* limit avoids an abrupt cruise
        # speed jump; the common acceptance limits remain 6 m/s² / 20 m/s³.
        # Preserve measured initial acceleration even when already above 1.
        generation_acceleration = min(6.0, max(1.0, abs(anchor.s_dd)))
        path, terminal = generator(anchor, desired, 3.0, 0.1, generation_acceleration)
        # Existing generator includes t0; only future 30 positions become actions.
        if path.s.shape != (31,) or not np.allclose(path.t, np.arange(31) * 0.1):
            raise ContractError("unexpected existing Frenet sample grid")
        if np.any(path.s < 0.0) or np.any(path.s > self._arc[-1]):
            raise ContractError(
                "teacher candidate exceeds measured route; no extrapolation"
            )
        r, _, _, n, _, _ = self.frame(path.s)
        xy = r + (path.d + lateral_offset)[:, None] * n
        proposal = e.proposal_from_world(xy[1:], token, model_version=self.version)
        from dataclasses import asdict

        return TeacherOutput(
            proposal,
            dict(
                adapter=self.version,
                legacy_generator=generator.__name__,
                initial_frenet=asdict(anchor),
                terminal_speed=asdict(terminal),
                generation_acceleration_limit=generation_acceleration,
                low_speed_lateral_offset=lateral_offset,
                stationary_hold_candidate=bool(stationary_hold),
                anchor_position_error_m=float(np.linalg.norm(xy[0] - ego.position)),
                reference="C2 interpolant of current map; unchanged measured corridor",
            ),
        )
