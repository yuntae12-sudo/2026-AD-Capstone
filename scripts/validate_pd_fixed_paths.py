#!/usr/bin/env python3
"""Production Hermite + PD on fixed references and real Waymax dynamics.

Synthetic controller gate only; full G3 also requires real road/collision cases.
No re-anchoring every tick, teacher, policy training, or relaxed thresholds.
"""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.contracts.planning import (
    ObservationToken,
    WaypointProposal,
    encode_waypoints,
    future_times,
)
from src.environment.womd_control.control import (
    Ego,
    PDMemory,
    Trajectory,
    pd_control,
    tracking_error,
)
from scripts.screen_womd_training import write_json


def reference(speed, acceleration, curvature, stop=False):
    def position(t):
        if stop:
            # Three-second endpoint-only stops can have a negative one-sided
            # knot derivative after sampling. Include a real stationary suffix.
            duration = 2.7
            s = np.clip(t / duration, 0, 1)
            distance = duration * speed * (s - s**3 + 0.5 * s**4)
        else:
            distance = speed * t + 0.5 * acceleration * t * t
        return (
            np.array(
                [
                    np.sin(curvature * distance) / curvature,
                    (1 - np.cos(curvature * distance)) / curvature,
                ]
            )
            if curvature
            else np.array([distance, 0.0])
        )

    return position


def run_case(
    name, speed, acceleration, curvature, stop=False, offset=0.0, heading_error=0.0
):
    import jax
    import jax.numpy as jnp
    from waymax import datatypes, dynamics

    read = reference(speed, acceleration, curvature, stop)
    xy = np.stack([read(t) for t in future_times(0)])
    position = np.zeros(2)
    velocity = np.array([speed, 0.0])
    a = np.array([acceleration, speed * speed * curvature])
    proposal = WaypointProposal(
        ObservationToken("synthetic-pd", name, 0, 0),
        1,
        name,
        0,
        position,
        0.0,
        velocity,
        a,
        True,
        future_times(0),
        xy,
        encode_waypoints(xy, position, 0.0, velocity, a),
    )
    trajectory = Trajectory(proposal)
    z = jnp.zeros((1, 1), jnp.float32)
    one = jnp.ones((1, 1), jnp.float32)
    state = datatypes.Trajectory(
        x=z,
        y=one * offset,
        z=z,
        yaw=one * heading_error,
        vel_x=one * speed,
        vel_y=z,
        length=one * 4.5,
        width=one * 1.8,
        height=one * 1.5,
        valid=jnp.ones((1, 1), bool),
        timestamp_micros=jnp.zeros((1, 1), jnp.int32),
    )
    step = jax.jit(
        dynamics.InvertibleBicycleModel(dt=0.1, normalize_actions=False).compute_update
    )
    memory = PDMemory(last_a=acceleration, last_kappa=curvature)
    rows = []
    for tick in range(30):
        ego = Ego(
            np.array([float(state.x[0, 0]), float(state.y[0, 0])]),
            np.array([float(state.vel_x[0, 0]), float(state.vel_y[0, 0])]),
            a,
            True,
            float(state.yaw[0, 0]),
            4.5,
            1.8,
        )
        control, memory = pd_control(trajectory, ego, memory, tick)
        applied = np.array(
            [control.acceleration_mps2, control.curvature_per_m], np.float32
        )
        memory = replace(memory, last_a=float(applied[0]), last_kappa=float(applied[1]))
        update = step(
            datatypes.Action(data=jnp.array([applied]), valid=jnp.ones((1, 1), bool)),
            state,
        )
        state = state.replace(
            x=update.x,
            y=update.y,
            yaw=update.yaw,
            vel_x=update.vel_x,
            vel_y=update.vel_y,
            valid=update.valid,
        )
        actual = replace(
            ego,
            position=np.array([float(state.x[0, 0]), float(state.y[0, 0])]),
            velocity=np.array([float(state.vel_x[0, 0]), float(state.vel_y[0, 0])]),
            yaw=float(state.yaw[0, 0]),
        )
        error = tracking_error(trajectory, actual, (tick + 1) * 0.1)
        rows.append(
            dict(
                position=error.position_m,
                speed=error.speed_mps,
                heading=error.heading_deg,
            )
        )
    stats = {}
    for key in ("position", "speed", "heading"):
        values = [r[key] for r in rows if r[key] is not None]
        stats[key + "_rms"] = (
            float(np.sqrt(np.mean(np.square(values)))) if values else None
        )
        stats[key + "_max"] = max(values, default=None)
    passed = (
        stats["position_rms"] <= 0.3
        and stats["position_max"] <= 0.75
        and stats["speed_rms"] <= 1
        and stats["speed_max"] <= 2
        and (stats["heading_max"] or 0) <= 15
    )
    return dict(
        case=name, passed=passed, **stats, final_speed_mps=actual.speed, steps=rows
    )


def run_real_fixed(proto):
    from src.environment.womd_control.environment import (
        SACPlanningEnvironment,
        route_reference_proposal,
    )
    from src.scenarios.womd20 import parse_scenario
    from scripts.screen_womd_training import development_split

    scenario = parse_scenario(proto.read_bytes())
    if development_split(scenario.scenario_id)[0] == "test":
        raise ValueError("held-out test scene")
    e = SACPlanningEnvironment()
    boundary = e.reset(scenario)
    proposal = route_reference_proposal(e, boundary.token)
    trajectory = Trajectory(proposal)
    check = e._check(trajectory, e._check_objects())
    if not check.valid:
        return dict(
            case=scenario.scenario_id,
            passed=False,
            initial_check=check.reason,
            steps=[],
        )
    rows = []
    for tick in range(30):
        control, memory = pd_control(trajectory, e.ego, e.memory, tick)
        result, boundary = e._advance(proposal, check, control, memory, trajectory)
        error = result.tracking_error
        rows.append(
            dict(
                position=error.position_m,
                speed=error.speed_mps,
                heading=error.heading_deg,
                event_flags=result.event_flags,
            )
        )
        if result.terminated or result.truncated:
            break
    stats = {}
    for key in ("position", "speed", "heading"):
        values = [r[key] for r in rows if r[key] is not None]
        stats[key + "_rms"] = (
            float(np.sqrt(np.mean(np.square(values)))) if values else None
        )
        stats[key + "_max"] = max(values, default=None)
    passed = (
        len(rows) == 30
        and stats["position_rms"] <= 0.3
        and stats["position_max"] <= 0.75
        and stats["speed_rms"] <= 1
        and stats["speed_max"] <= 2
        and (stats["heading_max"] or 0) <= 15
        and not any(r["event_flags"] for r in rows)
    )
    return dict(
        case=scenario.scenario_id,
        passed=passed,
        initial_check=check.reason,
        **stats,
        steps=rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sac_environment_gate/pd_fixed_paths.json"),
    )
    parser.add_argument("--real-proto", type=Path, action="append", default=[])
    args = parser.parse_args()
    cases = [
        ("straight", 5.0, 0.0, 0.0),
        ("accel_1", 5.0, 1.0, 0.0),
        ("accel_2", 5.0, 2.0, 0.0),
        ("brake", 10.0, -2.0, 0.0),
        ("curve_tight", 2.0, 0.0, 0.15),
        ("curve_medium", 5.0, 0.0, 0.05),
        ("curve_fast", 10.0, 0.0, 0.02),
        ("low_speed", 0.2, 0.0, 0.15),
        ("stationary", 0.0, 0.0, 0.0),
        ("stop_straight", 5.0, 0.0, 0.0, True),
        ("stop_curve", 5.0, 0.0, 0.02, True),
        ("lateral_error", 5.0, 0.0, 0.0, False, 0.3),
        ("heading_error", 5.0, 0.0, 0.0, False, 0.0, np.deg2rad(5)),
    ]
    report = dict(
        scope="synthetic production interpolation/PD fixed 3s reference; not complete G3 road gate",
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        result = run_case(*case)
        report["cases"].append(result)
        write_json(args.output, report)
        print({k: v for k, v in result.items() if k != "steps"}, flush=True)
    report["all_passed"] = all(c["passed"] for c in report["cases"])
    report["real_fixed_paths"] = [run_real_fixed(path) for path in args.real_proto]
    report["all_passed"] &= all(c["passed"] for c in report["real_fixed_paths"])
    write_json(args.output, report)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
