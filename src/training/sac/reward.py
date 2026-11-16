"""Shared KEEP/FOLLOW reward from actual post-step facts, v1.2 section 11."""

from dataclasses import dataclass
import numpy as np

from src.contracts.planning import Maneuver, Source, TerminalReason
from src.environment.womd_control.geometry import project, wrap


def collision_ttc(ego, objects, horizon=3.0):
    """Exact interval SAT for constant-velocity, constant-yaw actual OBBs."""
    earliest = float("inf")
    for other in objects:
        rotations = [
            np.array([[np.cos(o.yaw), -np.sin(o.yaw)], [np.sin(o.yaw), np.cos(o.yaw)]])
            for o in (ego, other)
        ]
        axes = np.concatenate([r.T for r in rotations])
        delta, velocity = other.position - ego.position, other.velocity - ego.velocity
        enter, leave = 0.0, horizon
        for axis in axes:
            radius = sum(
                abs(axis @ r[:, 0]) * o.length / 2 + abs(axis @ r[:, 1]) * o.width / 2
                for r, o in zip(rotations, (ego, other))
            )
            position, speed = float(delta @ axis), float(velocity @ axis)
            if abs(speed) < 1e-12:
                if abs(position) > radius:
                    leave = -1.0
                    break
            else:
                a, b = sorted(
                    ((-radius - position) / speed, (radius - position) / speed)
                )
                enter, leave = max(enter, a), min(leave, b)
        if enter <= leave:
            earliest = min(earliest, enter)
    return earliest


@dataclass(frozen=True)
class RewardResult:
    terms: dict
    step_unclipped: float
    step: float
    event: float
    training: float
    clipped: bool
    flags: tuple
    sample_valid: bool


def calculate_reward(
    command,
    *,
    delta_s,
    speed,
    speed_limit,
    gap,
    ttc,
    acceleration,
    jerk,
    lateral_acceleration,
    tracking_error,
    raw_rejected,
    source,
    terminal_reason,
    flags=(),
    command_violation=False
):
    if terminal_reason == TerminalReason.DATA_CORRUPTION:
        return RewardResult({}, 0.0, 0.0, 0.0, 0.0, False, tuple(flags), False)
    c = lambda z: float(np.clip(z, 0.0, 1.0))
    effective = min(command.desired_speed_mps, speed_limit, 15.0)
    terms = {"progress": 0.5 * c(delta_s / max(effective * 0.1, 0.1))}
    if command.maneuver == Maneuver.KEEP:
        terms["speed"] = -0.2 * c(abs(speed - effective) / max(effective, 1.0))
    elif command.maneuver == Maneuver.FOLLOW:
        terms["gap"] = (
            -0.5
            * c(abs(gap - command.minimum_gap_m - command.time_gap_s * speed) / 10.0)
            if gap is not None
            else 0.0
        )
    else:
        raise ValueError("reward for this behavior is not gated")
    terms.update(
        ttc=-0.2 * c((3.0 - ttc) / 3.0),
        comfort=-0.05
        * (
            c(abs(acceleration) / 2.0)
            + (c(abs(jerk) / 3.0) if jerk is not None else 0.0)
            + c(abs(lateral_acceleration) / 2.0)
        ),
        tracking=-0.1 * c(tracking_error / 0.5) if tracking_error is not None else 0.0,
        raw_reject=-0.5 * bool(raw_rejected),
        failure_fallback=-0.5
        * (source in (Source.REUSED, Source.BRAKE, Source.EMERGENCY)),
        command_violation=-0.5 * bool(command_violation),
    )
    failures = {
        TerminalReason.COLLISION,
        TerminalReason.OFFROAD,
        TerminalReason.CORRIDOR_VIOLATION,
        TerminalReason.MAP_COVERAGE_EXIT,
    }
    if terminal_reason in failures or any(
        f in {r.value for r in failures} for f in flags
    ):
        event = -2000.0
    elif terminal_reason in (TerminalReason.ROUTE_GOAL, TerminalReason.STOP_GOAL):
        event = 10.0
    elif terminal_reason == TerminalReason.TASK_LOST:
        event = -5.0
    else:
        event = 0.0
    raw = float(sum(terms.values()))
    step = float(np.clip(raw, -5.0, 2.0))
    return RewardResult(
        terms, raw, step, event, (step + event) / 100.0, raw != step, tuple(flags), True
    )


def reward_after_step(environment, result, before, previous, before_progress):
    e = environment
    if not result.sample_valid:
        return RewardResult({}, 0.0, 0.0, 0.0, 0.0, False, ("DATA_CORRUPTION",), False)
    ego = e.ego
    draft = result.executed_command_snapshot.draft
    acceleration = (ego.speed - before.speed) / 0.1
    jerk = (
        (acceleration - (before.speed - previous.speed) / 0.1) / 0.1
        if before.acceleration_valid
        else None
    )
    gap = None
    if (
        draft.maneuver == Maneuver.FOLLOW
        and e.target_cache is not None
        and e.tick - e.target_last_tick <= 2
    ):
        target = e.target_cache
        p = target.position + target.velocity * (e.tick - e.target_last_tick) * 0.1
        gap = (
            project(e.route.center, p)[0]
            - project(e.route.center, ego.position)[0]
            - (ego.length + target.length) / 2
        )
    return calculate_reward(
        draft,
        delta_s=e.feedback.progress - before_progress,
        speed=ego.speed,
        speed_limit=e.route.speed_limit(ego.position),
        gap=gap,
        ttc=collision_ttc(ego, e._objects().values()),
        acceleration=acceleration,
        jerk=jerk,
        lateral_acceleration=0.5
        * (ego.speed + before.speed)
        * wrap(ego.yaw - before.yaw)
        / 0.1,
        tracking_error=(
            result.tracking_error.position_m if result.tracking_error else None
        ),
        raw_rejected=not result.check_result.valid,
        source=result.source,
        terminal_reason=result.terminal_reason,
        flags=getattr(result, "event_flags", ()),
        command_violation=result.check_result.reason == "COMMAND_VIOLATION",
    )
