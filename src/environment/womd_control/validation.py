"""Measured rollout diagnostics, distinct from hypothetical proposal checks."""

from collections import Counter
from dataclasses import asdict
import time

import numpy as np

from src.contracts.planning import DT, to_wire
from .geometry import wrap


def step_record(environment, result, before, previous, previous_control, duration_s):
    after = environment.ego
    acceleration = (after.speed - before.speed) / DT
    previous_acceleration = (before.speed - previous.speed) / DT
    yaw_rate = wrap(after.yaw - before.yaw) / DT
    motion = {
        "acceleration_mps2": acceleration,
        "jerk_mps3": (acceleration - previous_acceleration) / DT,
        "yaw_rate_rps": yaw_rate,
        "lateral_acceleration_mps2": 0.5 * (after.speed + before.speed) * yaw_rate,
        "input_jerk_mps3": (
            (result.actual_control.acceleration_mps2 - previous_control.last_a) / DT
            if previous_control.applied_control_valid
            else None
        ),
        "input_curvature_rate": (
            (result.actual_control.curvature_per_m - previous_control.last_kappa) / DT
            if previous_control.applied_control_valid
            else None
        ),
    }
    return {
        "tick": result.post_tick,
        "source_frame": environment.frame,
        "source": result.source.value,
        "check": asdict(result.check_result),
        "attempts": [
            {**asdict(attempt), "source": attempt.source.value}
            for attempt in result.attempts
        ],
        "tracking_error": (
            asdict(result.tracking_error) if result.tracking_error else None
        ),
        "control": to_wire(result.actual_control),
        "world_xy": after.position.tolist(),
        "speed_mps": after.speed,
        "motion": {
            name: float(value) if value is not None and np.isfinite(value) else None
            for name, value in motion.items()
        },
        "route_progress_m": environment.feedback.progress,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "sample_valid": result.sample_valid,
        "terminal_reason": (
            result.terminal_reason.value if result.terminal_reason else None
        ),
        "waymax_offroad": environment.waymax_offroad,
        "waymax_overlap": environment.waymax_overlap,
        "metrics_valid": environment.metrics_valid,
        "source_clock_drift_s": environment.source_clock_drift_s,
        "clipped_features": environment.clipped_features,
        "wall_time_s": duration_s,
    }


def tracking_summary(rows):
    measured = [
        row["tracking_error"] for row in rows if row.get("tracking_error") is not None
    ]
    result = {"samples": len(measured), "without_reference": len(rows) - len(measured)}
    for name in ("position_m", "speed_mps", "heading_deg"):
        values = [item[name] for item in measured if item[name] is not None]
        result[name] = {
            "samples": len(values),
            "rms": float(np.sqrt(np.mean(np.square(values)))) if values else None,
            "max": float(max(values)) if values else None,
        }
    return result


def summarize(report):
    rows = report["steps"]
    sources = Counter(row["source"] for row in rows)
    last = rows[-1] if rows else {}
    reason = last.get("terminal_reason")
    ended = bool(last.get("terminated") or last.get("truncated"))
    outcome = (
        reason
        if reason in ("ROUTE_GOAL", "RECORD_END", "TIME_LIMIT")
        else ("FAILURE" if ended else "STEP_LIMIT_REACHED")
    )
    if report.get("status") in ("RUNTIME_ERROR", "RESET_REJECTED"):
        outcome = report["status"]
    attempts = [attempt for row in rows for attempt in row.get("attempts", [])]
    report.update(
        status=(
            report.get("status")
            if outcome in ("RUNTIME_ERROR", "RESET_REJECTED")
            else "EPISODE_ENDED" if ended else "STEP_LIMIT_REACHED"
        ),
        outcome=outcome,
        terminal_reason=reason,
        episode_ended=ended,
        steps_executed=len(rows),
        simulated_duration_s=len(rows) * DT,
        execution_sources=dict(sources),
        check_reasons=dict(Counter(row["check"]["reason"] for row in rows)),
        rejected_candidates=dict(
            Counter(
                f"{a['source']}:{a['check']['reason']}"
                for a in attempts
                if not a["check"]["valid"]
            )
        ),
        rejection_details=dict(
            Counter(
                f"{a['source']}:{violation}"
                for a in attempts
                for violation in a["check"]["violations"]
            )
        ),
        source_transitions=dict(
            Counter(
                f"{a['source']}->{b['source']}"
                for a, b in zip(rows, rows[1:])
                if a["source"] != b["source"]
            )
        ),
        tracking=tracking_summary(rows),
        tracking_by_source={
            source: tracking_summary([r for r in rows if r["source"] == source])
            for source in sources
        },
        stopped_ticks=sum(row["speed_mps"] <= 0.2 for row in rows),
        final_progress_m=last.get("route_progress_m"),
        timing={
            "total_wall_s": sum(row["wall_time_s"] for row in rows),
            "p95_step_wall_s": (
                float(np.percentile([row["wall_time_s"] for row in rows], 95))
                if rows
                else None
            ),
        },
    )
    return report


def measured_step(environment, boundary, proposer):
    before = environment.ego
    previous = environment.ego_at(environment.frame - 1)
    previous_control = environment.memory
    started = time.perf_counter()
    result, boundary = environment.step_waypoints(
        proposer(environment, boundary.token), boundary.token
    )
    row = step_record(
        environment,
        result,
        before,
        previous,
        previous_control,
        time.perf_counter() - started,
    )
    return result, boundary, row
