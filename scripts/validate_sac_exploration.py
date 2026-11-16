#!/usr/bin/env python3
"""4096 per case/sigma using the actual sampler and production kinematic checker.

No road, objects, PD or learned-policy performance is certified by this gate.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jnp
from src.contracts.planning import (
    ObservationToken,
    WaypointProposal,
    encode_waypoints,
    decode_waypoints,
    future_times,
)
from src.environment.womd_control.control import (
    Trajectory,
    CheckResult,
    _reference_kinematics,
)
from src.training.sac.networks import sample_action
from src.training.sac.runtime import versions, write_report
from src.training.sac.checkpoint import file_hash

CASES = (
    (10.0, 0.0, 0.0),
    (5.0, 1.0, 0.0),
    (5.0, 2.0, 0.0),
    (10.0, -2.0, 0.0),
    (5.0, 0.0, 0.05),
    (10.0, 0.0, 0.02),
    (2.0, 0.0, 0.15),
    (1.0, 0.0, 0.15),
    (0.2, 0.0, 0.15),
)


def kinematic_reason(proposal, curvature):
    checked = _reference_kinematics(Trajectory(proposal), 0)
    if isinstance(checked, CheckResult):
        return checked.reason + ":" + ",".join(checked.violations)
    _, _, points, v, s = checked
    tangent = np.column_stack([1 - curvature * points[:, 1], curvature * points[:, 0]])
    return (
        "speed_limit"
        if np.any(s > 15.0 + 1e-6)
        else "reverse_velocity" if np.any(np.sum(tangent * v, axis=1) < -0.01) else "OK"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    version = versions(args.manifest)
    count = 16 if args.smoke else 4096
    report = dict(
        scope="SYNTHETIC_KINEMATICS",
        versions=version,
        runner_sha256=file_hash(__file__),
        seed=20261116,
        requested_samples=count,
        cases=[],
        status="RUNNING",
        qualified=False,
    )
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if (
            report["versions"] != version
            or report["requested_samples"] != count
            or report["runner_sha256"] != file_hash(__file__)
        ):
            raise ValueError("incompatible exploration resume")
    key = jax.random.PRNGKey(report["seed"])
    for case_index, (speed, accel, curvature) in enumerate(CASES):
        case = f"v{speed}_a{accel}_k{curvature}"
        times = future_times(0)
        distance = speed * times + 0.5 * accel * times**2
        xy = (
            np.column_stack(
                [
                    np.sin(curvature * distance) / curvature,
                    (1 - np.cos(curvature * distance)) / curvature,
                ]
            )
            if curvature
            else np.column_stack([distance, np.zeros(30)])
        )
        velocity = np.array([speed, 0.0])
        acceleration = np.array([accel, speed * speed * curvature])
        position = np.zeros(2)
        mu = encode_waypoints(xy, position, 0.0, velocity, acceleration).astype(
            np.float32
        )
        token = ObservationToken("synthetic-exploration", case, 0, 0)
        mean_raw = mu.astype(np.float64)
        mean_proposal = WaypointProposal(
            token,
            1,
            f"{case}:mean",
            0,
            position,
            0.0,
            velocity,
            acceleration,
            True,
            times,
            decode_waypoints(mean_raw, position, 0.0, velocity, acceleration),
            mean_raw,
        )
        mean_reason = kinematic_reason(mean_proposal, curvature)
        for sigma_index, sigma in enumerate((0.02, 0.03)):
            index = case_index * 2 + sigma_index
            if len(report["cases"]) <= index:
                report["cases"].append(
                    dict(
                        case=case,
                        sigma=sigma,
                        speed=speed,
                        acceleration=accel,
                        curvature=curvature,
                        mean_valid=mean_reason == "OK",
                        mean_reason=mean_reason,
                        samples=0,
                        accepted=0,
                        reasons={},
                    )
                )
            row = report["cases"][index]
            if row["samples"] == count:
                continue
            rho = jnp.full(
                (count, 2), np.log(1.5) if sigma == 0.02 else 100.0, jnp.float32
            )
            actions = np.asarray(
                sample_action(
                    jnp.asarray(np.broadcast_to(mu, (count, 30, 2))),
                    rho,
                    jnp.full(count, speed, jnp.float64),
                    jax.random.fold_in(key, index),
                )[0]
            )
            reasons = Counter(row["reasons"])
            for i in range(row["samples"], count):
                raw = actions[i]
                points = decode_waypoints(raw, position, 0.0, velocity, acceleration)
                proposal = WaypointProposal(
                    token,
                    1,
                    f"{case}:{sigma}:{i}",
                    0,
                    position,
                    0.0,
                    velocity,
                    acceleration,
                    True,
                    times,
                    points,
                    raw,
                )
                reason = kinematic_reason(proposal, curvature)
                reasons[reason] += 1
                row["samples"] += 1
                row["accepted"] += int(reason == "OK")
                if row["samples"] % 128 == 0 or row["samples"] == count:
                    row["reasons"] = dict(reasons)
                    write_report(args.output, report)
                    print(json.dumps(row), flush=True)
    report.update(
        status="COMPLETE",
        qualified=not args.smoke,
        passed=all(
            r["mean_valid"] and r["samples"] == 4096 and r["accepted"] / 4096 >= 0.95
            for r in report["cases"]
        ),
    )
    write_report(args.output, report)
    return 0 if report["passed"] or args.smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())
