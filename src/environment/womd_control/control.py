"""Quintic interpolation, common PD and conservative finite-horizon checks.

No feedforward and no learned controller. Sweeps are discrete numerical checks,
not a continuous safety certificate. The virtual model matches Waymax's
InvertibleBicycleModel equations, including independent initial vx/vy and yaw.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.contracts.planning import ControlCommand, DT, Source, WaypointProposal
from .geometry import box, overlaps, project, wrap


@dataclass(frozen=True)
class Ego:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    acceleration_valid: bool
    yaw: float
    length: float
    width: float

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))


@dataclass(frozen=True)
class PDMemory:
    previous_e_y: float = 0.0
    previous_e_v: float = 0.0
    filtered_de_y: float = 0.0
    filtered_de_v: float = 0.0
    last_a: float = 0.0
    last_kappa: float = 0.0
    errors_valid: bool = False
    applied_control_valid: bool = False


@dataclass(frozen=True)
class CheckResult:
    valid: bool
    reason: str
    checked_samples: int = 0
    max_position_error_m: float = 0.0
    max_speed_error_mps: float = 0.0


class Trajectory:
    def __init__(self, proposal: WaypointProposal):
        self.proposal = proposal
        self.start = proposal.generated_tick * DT
        self.end = self.start + 3.0
        p = np.vstack([proposal.anchor_world_xy, proposal.xy_world]).copy()
        v, a = np.zeros_like(p), np.zeros_like(p)
        v[0], a[0] = proposal.anchor_velocity_world, proposal.anchor_acceleration_world
        v[1:30] = (p[2:] - p[:-2]) / (2 * DT)
        a[1:30] = (p[2:] - 2 * p[1:30] + p[:-2]) / DT**2
        v[30] = (3 * p[30] - 4 * p[29] + p[28]) / (2 * DT)
        a[30] = (p[30] - 2 * p[29] + p[28]) / DT**2
        # Only a terminal near-stationary suffix may be aligned (<= 1 mm).
        self.edited_indices = ()
        self.max_edit_m = 0.0
        for first in range(1, 29):
            if np.max(np.linalg.norm(p[first:] - p[first], axis=1)) <= 0.001:
                old = p[first:].copy()
                if (
                    first == 1
                    and np.max(np.linalg.norm(p[1:] - p[0], axis=1)) <= 0.001
                    and np.linalg.norm(v[0]) <= 0.001
                    and np.linalg.norm(a[0]) <= 0.01
                ):
                    p[first:] = p[0]
                else:
                    p[first:] = p[first]
                self.max_edit_m = float(np.max(np.linalg.norm(old - p[first:], axis=1)))
                self.edited_indices = tuple(range(first, 31))
                # Recompute the neighboring finite difference after editing.
                v[1:30] = (p[2:] - p[:-2]) / (2 * DT)
                a[1:30] = (p[2:] - 2 * p[1:30] + p[:-2]) / DT**2
                v[first:], a[first:] = 0, 0
                break
        c = np.zeros((30, 2, 6), np.float64)
        c[:, :, 0], c[:, :, 1], c[:, :, 2] = p[:-1], v[:-1] * DT, a[:-1] * DT**2 / 2
        rhs = np.stack(
            [
                p[1:] - p[:-1] - v[:-1] * DT - a[:-1] * DT**2 / 2,
                (v[1:] - v[:-1]) * DT - a[:-1] * DT**2,
                (a[1:] - a[:-1]) * DT**2,
            ],
            axis=-1,
        )
        inverse = np.linalg.inv(np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], float))
        c[:, :, 3:] = rhs @ inverse.T
        self.coefficients = c

    def evaluate(self, times, derivative=0):
        times = np.asarray(times, np.float64)
        if np.any(times < self.start - 1e-9) or np.any(times > self.end + 1e-9):
            raise ValueError("trajectory extrapolation forbidden")
        t = np.clip((times - self.start) / DT, 0, 30)
        i = np.minimum(np.floor(t).astype(int), 29)
        z = t - i
        coefficients = self.coefficients[i].copy()
        for _ in range(derivative):
            coefficients = (
                coefficients[..., 1:] * np.arange(1, coefficients.shape[-1]) / DT
            )
        return np.sum(
            coefficients * z[..., None, None] ** np.arange(coefficients.shape[-1]),
            axis=-1,
        )

    def yaw(self, time):
        velocity = self.evaluate(time, 1)
        if np.linalg.norm(velocity) > 1e-9:
            return float(np.arctan2(velocity[1], velocity[0]))
        for t in np.arange(time, self.start - 1e-9, -DT):
            v = self.evaluate(max(t, self.start), 1)
            if np.linalg.norm(v) > 1e-9:
                return float(np.arctan2(v[1], v[0]))
        return self.proposal.anchor_yaw


def advance(ego, acceleration, curvature, duration=DT):
    distance = ego.speed * duration + 0.5 * acceleration * duration**2
    yaw = wrap(ego.yaw + curvature * distance)
    heading = np.array([np.cos(ego.yaw), np.sin(ego.yaw)])
    position = (
        ego.position
        + ego.velocity * duration
        + 0.5 * acceleration * heading * duration**2
    )
    velocity = (ego.speed + acceleration * duration) * np.array(
        [np.cos(yaw), np.sin(yaw)]
    )
    return Ego(
        position,
        velocity,
        (velocity - ego.velocity) / max(duration, 1e-12),
        True,
        float(yaw),
        ego.length,
        ego.width,
    )


def pd_control(trajectory, ego, memory, tick, source=Source.NEW):
    now = tick * DT
    v_ref = np.linalg.norm(trajectory.evaluate(min(now + DT, trajectory.end), 1))
    preview = np.clip(2 / max(ego.speed, 1), 0.3, 1.0)
    target_time = min(now + preview, trajectory.end)
    delta = trajectory.evaluate(target_time) - ego.position
    length = max(float(np.linalg.norm(delta)), 1.0)
    e_y = -np.sin(ego.yaw) * delta[0] + np.cos(ego.yaw) * delta[1]
    e_v = float(v_ref - ego.speed)
    e_heading = wrap(trajectory.yaw(target_time) - ego.yaw)
    de_y = (
        0.5 * memory.filtered_de_y + 0.5 * (e_y - memory.previous_e_y) / DT
        if memory.errors_valid
        else 0.0
    )
    de_v = (
        0.5 * memory.filtered_de_v + 0.5 * (e_v - memory.previous_e_v) / DT
        if memory.errors_valid
        else 0.0
    )
    a_raw = 6 * e_v + 0.1 * de_v
    k_raw = (
        e_y / length**2
        + 0.5 * e_heading / length
        + 0.1 * de_y / (max(ego.speed, 1) * length)
    )
    a_jerk = np.clip(a_raw, memory.last_a - 2, memory.last_a + 2)
    a = float(np.clip(a_jerk, max(-6, -ego.speed / DT), 6))
    k = float(
        np.clip(
            np.clip(k_raw, memory.last_kappa - 0.05, memory.last_kappa + 0.05),
            -0.3,
            0.3,
        )
    )
    reason = (
        "NONREVERSE_OVERRIDES_JERK"
        if a < memory.last_a - 2 - 1e-9 or a > memory.last_a + 2 + 1e-9
        else None
    )
    control = ControlCommand(
        tick, a, k, source, bool(abs(a - a_raw) > 1e-9 or abs(k - k_raw) > 1e-9), reason
    )
    updated = PDMemory(float(e_y), e_v, float(de_y), float(de_v), a, k, True, True)
    return control, updated


def emergency(ego, memory, tick):
    a = -min(6.0, ego.speed / DT)
    k = memory.last_kappa if ego.speed > 1e-9 else 0.0
    control = ControlCommand(tick, a, k, Source.EMERGENCY, True, "EMERGENCY_BRAKING")
    return control, replace(
        memory, last_a=a, last_kappa=k, errors_valid=False, applied_control_valid=True
    )


def _scene_check(ego, route, objects, elapsed):
    if not route.contains(ego.position, ego.yaw, ego.length, ego.width, 0.1):
        return "PREDICTED_OFFROAD"
    ego_box = box(ego.position, ego.yaw, ego.length, ego.width, 0.3)
    for other in objects:
        position = other.position + other.velocity * elapsed
        radius = 0.5 * (
            np.hypot(ego.length + 0.6, ego.width + 0.6)
            + np.hypot(other.length, other.width)
        )
        if np.linalg.norm(position - ego.position) <= radius and overlaps(
            ego_box, box(position, other.yaw, other.length, other.width)
        ):
            return "PREDICTED_COLLISION"
    return None


def check_trajectory(trajectory, ego, route, objects, memory, tick):
    now = tick * DT
    if now + DT > trajectory.end + 1e-9:
        return CheckResult(False, "OLD_PLAN_EXPIRED")
    probe = np.arange(now, trajectory.end + 0.001, 0.02)
    # Include interior extrema of speed^2, not only the regular time grid.
    extrema = []
    for index, c in enumerate(trajectory.coefficients):
        vx = c[0, 1:] * np.arange(1, 6) / DT
        vy = c[1, 1:] * np.arange(1, 6) / DT
        speed2 = np.polynomial.polynomial.polyadd(
            np.polynomial.polynomial.polymul(vx, vx),
            np.polynomial.polynomial.polymul(vy, vy),
        )
        for root in np.polynomial.polynomial.polyroots(
            np.polynomial.polynomial.polyder(speed2)
        ):
            if abs(root.imag) < 1e-8 and 0 < root.real < 1:
                t = trajectory.start + (index + root.real) * DT
                if now <= t <= trajectory.end:
                    extrema.append(t)
    probe = np.unique(np.r_[probe, extrema])
    max_speed = max(
        15.0, float(np.linalg.norm(trajectory.evaluate(probe, 1), axis=-1).max())
    )
    max_other = max((o.speed for o in objects), default=0.0)
    interval = min(
        0.02,
        0.1 / max(max_speed + max_other, 1),
        np.deg2rad(2) / max(0.3 * max_speed, 1e-9),
    )
    if interval < 0.002 - 1e-12:
        return CheckResult(False, "UNRESOLVED_SWEEP")
    times = np.linspace(
        now, trajectory.end, int(np.ceil((trajectory.end - now) / interval)) + 1
    )
    # Include both sides of every knot; jerk can be discontinuous for C2 paths.
    knots = (
        np.arange(max(1, int(round((now - trajectory.start) / DT))), 30) * DT
        + trajectory.start
    )
    extras = np.concatenate([knots - 1e-8, knots, knots + 1e-8])
    times = np.unique(
        np.r_[times, extrema, extras[(extras >= now) & (extras <= trajectory.end)]]
    )
    position = trajectory.evaluate(times)
    v, a, j = [trajectory.evaluate(times, d) for d in (1, 2, 3)]
    speed = np.linalg.norm(v, axis=1)
    moving = speed >= 0.01
    denom = np.maximum(speed, 0.01)
    cross_va = v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]
    dot_va = np.sum(v * a, axis=1)
    at = dot_va / denom
    alat = cross_va / denom
    kappa = cross_va / denom**3
    jerk = (
        np.sum(a * a, axis=1) + np.sum(v * j, axis=1)
    ) / denom - dot_va**2 / denom**3
    krate = (
        v[:, 0] * j[:, 1] - v[:, 1] * j[:, 0]
    ) / denom**3 - 3 * cross_va * dot_va / denom**5
    if (
        np.any(np.abs(at[moving]) > 6 + 1e-6)
        or np.any(np.abs(alat[moving]) > 4 + 1e-6)
        or np.any(np.abs(kappa[moving]) > 0.3 + 1e-6)
        or np.any(np.abs(jerk[moving]) > 20 + 1e-6)
        or np.any(np.abs(krate[moving]) > 0.5 + 1e-6)
        or np.any(np.linalg.norm(a[~moving], axis=1) > 6 + 1e-6)
        or np.any(np.linalg.norm(j[~moving], axis=1) > 20 + 1e-6)
    ):
        return CheckResult(False, "KINEMATIC_LIMIT", len(times))
    for t, p, velocity, s in zip(times, position, v, speed):
        _, _, tangent = project(route.center, p)
        if (
            s > min(route.speed_limit(p), 15) + 1e-6
            or np.dot(tangent, velocity) < -0.01
        ):
            return CheckResult(False, "KINEMATIC_LIMIT", len(times))
        pose = replace(ego, position=p, velocity=velocity, yaw=trajectory.yaw(t))
        reason = _scene_check(pose, route, objects, t - now)
        if reason:
            return CheckResult(False, reason, len(times))
    # Virtual PD uses private memory and the same discrete dynamics as Waymax.
    virtual, virtual_memory = ego, memory
    max_position_error = max_speed_error = 0.0
    count = int(round((trajectory.end - now) / DT))
    for step in range(count):
        control, virtual_memory = pd_control(
            trajectory, virtual, virtual_memory, tick + step
        )
        next_virtual = advance(
            virtual, control.acceleration_mps2, control.curvature_per_m
        )
        lateral_actual = (
            0.5
            * (virtual.speed + next_virtual.speed)
            * wrap(next_virtual.yaw - virtual.yaw)
            / DT
        )
        if abs(lateral_actual) > 4 + 1e-6:
            return CheckResult(
                False,
                "KINEMATIC_LIMIT",
                len(times),
                max_position_error,
                max_speed_error,
            )
        for duration in np.linspace(0, DT, int(np.ceil(DT / interval)) + 1)[1:]:
            point = advance(
                virtual,
                control.acceleration_mps2,
                control.curvature_per_m,
                float(duration),
            )
            time = now + step * DT + duration
            if point.speed > min(15.0, route.speed_limit(point.position)) + 1e-6:
                return CheckResult(
                    False,
                    "KINEMATIC_LIMIT",
                    len(times),
                    max_position_error,
                    max_speed_error,
                )
            pe = float(np.linalg.norm(point.position - trajectory.evaluate(time)))
            ve = abs(point.speed - float(np.linalg.norm(trajectory.evaluate(time, 1))))
            max_position_error, max_speed_error = max(max_position_error, pe), max(
                max_speed_error, ve
            )
            if (
                pe > 0.75 + 1e-6
                or ve > 2 + 1e-6
                or (
                    point.speed >= 0.01
                    and abs(wrap(point.yaw - trajectory.yaw(time))) > np.pi / 12 + 1e-6
                )
            ):
                return CheckResult(
                    False,
                    "TRACKING_FAILURE",
                    len(times),
                    max_position_error,
                    max_speed_error,
                )
            reason = _scene_check(point, route, objects, time - now)
            if reason:
                return CheckResult(
                    False, reason, len(times), max_position_error, max_speed_error
                )
        virtual = advance(virtual, control.acceleration_mps2, control.curvature_per_m)
    return CheckResult(True, "OK", len(times), max_position_error, max_speed_error)
