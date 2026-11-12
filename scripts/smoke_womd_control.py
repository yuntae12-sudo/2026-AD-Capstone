#!/usr/bin/env python3
"""Run bounded, real control steps on a WOMD Scenario; never launch training."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    scripted_proposal,
)
from src.environment.womd_control.geometry import ResetRejected
from src.contracts.planning import to_wire


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--steps", type=positive, default=10)
    parser.add_argument(
        "--provider", choices=("fixed_keep", "fixed_follow"), default="fixed_keep"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/womd_control"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.shard.name}-record-{args.record_index}.json"
    report = {
        "scope": "bounded control smoke; not training, full G3/G4 acceptance or policy performance",
        "source_shard": args.shard.name,
        "record_index": args.record_index,
        "provider": args.provider,
        "planner": "constant-acceleration development input; not ASMC",
        "steps": [],
    }
    environment = SACPlanningEnvironment()
    try:
        boundary = environment.reset(
            (args.shard, args.record_index), seed=0, provider_spec=args.provider
        )
    except ResetRejected as error:
        report.update(status="RESET_REJECTED", reason=str(error))
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "status": "RESET_REJECTED",
                    "reason": str(error),
                    "report": str(output),
                }
            )
        )
        return 2
    report.update(
        scene_id=boundary.token.scene_id,
        start_frame=environment.start_frame,
        route_lane_ids=environment.route.lane_ids,
        boundary_feature_ids=environment.route.boundary_ids,
        geometry="boundary-reference projection with contiguous coverage; no nominal width",
        initial_world_xy=environment.ego.position.tolist(),
        goal_route_s=environment.route.goal_s,
    )
    for _ in range(args.steps):
        result, boundary = environment.step_waypoints(
            scripted_proposal(environment, boundary.token), boundary.token
        )
        report["steps"].append(
            {
                "tick": result.post_tick,
                "source_frame": environment.frame,
                "source": result.source.value,
                "check": asdict(result.check_result),
                "control": to_wire(result.actual_control),
                "world_xy": environment.ego.position.tolist(),
                "speed_mps": environment.ego.speed,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "terminal_reason": (
                    result.terminal_reason.value if result.terminal_reason else None
                ),
                "waymax_offroad": environment.waymax_offroad,
                "waymax_overlap": environment.waymax_overlap,
                "metrics_valid": environment.metrics_valid,
                "source_clock_drift_s": environment.source_clock_drift_s,
                "clipped_features": environment.clipped_features,
            }
        )
        if result.terminated or result.truncated:
            break
    report.update(
        status="CONTROL_STEPS_EXECUTED",
        steps_executed=len(report["steps"]),
        execution_sources=dict(Counter(step["source"] for step in report["steps"])),
        check_reasons=dict(
            Counter(step["check"]["reason"] for step in report["steps"])
        ),
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "scene_id",
                    "steps_executed",
                    "execution_sources",
                    "check_reasons",
                )
            }
        )
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
