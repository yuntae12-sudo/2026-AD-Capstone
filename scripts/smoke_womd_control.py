#!/usr/bin/env python3
"""Measure bounded or full-record WOMD control, without launching training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    scripted_proposal,
    route_reference_proposal,
)
from src.environment.womd_control.geometry import ResetRejected
from src.environment.womd_control.validation import measured_step, summarize


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def positive(value):
    value = nonnegative(value)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--record-index", type=nonnegative, default=0)
    parser.add_argument("--steps", type=positive, default=10)
    parser.add_argument(
        "--provider", choices=("fixed_keep", "fixed_follow"), default="fixed_keep"
    )
    parser.add_argument(
        "--planner",
        choices=("constant_acceleration", "route_reference"),
        default="constant_acceleration",
    )
    parser.add_argument("--progress-every", type=nonnegative, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/womd_control"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = (
        args.output_dir
        / f"{args.shard.name}-record-{args.record_index}-{args.provider}-{args.planner}-{args.steps}steps.json"
    )
    report = {
        "report_schema_version": 2,
        "scope": "measured control rollout; not training, full G3/G4 acceptance or policy performance",
        "source_shard": args.shard.name,
        "record_index": args.record_index,
        "provider": args.provider,
        "requested_steps": args.steps,
        "planner": args.planner,
        "planner_scope": "development input; not ASMC or learned policy",
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "status": "RUNNING",
        "steps": [],
    }

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    environment = SACPlanningEnvironment()
    proposer = (
        scripted_proposal
        if args.planner == "constant_acceleration"
        else route_reference_proposal
    )
    save()
    try:
        boundary = environment.reset(
            (args.shard, args.record_index), seed=0, provider_spec=args.provider
        )
        report.update(
            scene_id=boundary.token.scene_id,
            start_frame=environment.start_frame,
            last_source_frame=environment.last_frame,
            route_lane_ids=environment.route.lane_ids,
            boundary_feature_ids=environment.route.boundary_ids,
            geometry="boundary-reference projection with contiguous coverage; no nominal width",
            initial_world_xy=environment.ego.position.tolist(),
            goal_route_s=environment.route.goal_s,
        )
        for _ in range(args.steps):
            result, boundary, row = measured_step(environment, boundary, proposer)
            report["steps"].append(row)
            save()
            if args.progress_every and result.post_tick % args.progress_every == 0:
                print(
                    f"tick={result.post_tick} source={result.source.value} check={result.check_result.reason} speed={row['speed_mps']:.3f}",
                    flush=True,
                )
            if result.terminated or result.truncated:
                break
    except ResetRejected as error:
        report.update(status="RESET_REJECTED", reason=str(error))
    except BaseException as error:
        report.update(
            status="RUNTIME_ERROR", error_type=type(error).__name__, reason=str(error)
        )
        raise
    finally:
        summarize(report)
        save()
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "outcome",
                    "terminal_reason",
                    "steps_executed",
                    "execution_sources",
                    "check_reasons",
                )
            }
        )
    )
    print(output)
    return 2 if report["status"] == "RESET_REJECTED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
