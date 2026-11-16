#!/usr/bin/env python3
"""Time unchanged control loops with cold/warm resets and explicit device sync.

Run independently of screening/other CPU jobs. Timings include proposal, full
checker, actual Waymax dynamics, and next observation; exclude loading/rendering
and report serialization. This is not learner/teacher throughput.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    route_reference_proposal,
    scripted_proposal,
)
from src.scenarios.womd20 import parse_scenario
from scripts.screen_womd_training import development_split, write_json


def timing(values):
    return dict(
        samples=len(values),
        total_s=sum(values),
        mean_s=float(np.mean(values)) if values else None,
        p50_s=float(np.median(values)) if values else None,
        p95_s=float(np.percentile(values, 95)) if values else None,
        ticks_per_s=len(values) / sum(values) if values and sum(values) else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider", choices=("fixed_keep", "fixed_follow"), default="fixed_keep"
    )
    parser.add_argument(
        "--planner",
        choices=("constant_acceleration", "route_reference"),
        default="route_reference",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.trials) < 1:
        parser.error("steps/trials must be positive")
    import jax
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    raw = args.proto.read_bytes()
    scenario = parse_scenario(raw)
    split, _ = development_split(scenario.scenario_id)
    if split == "test":
        parser.error("Timing runs must not consume held-out test scenes")
    e = SACPlanningEnvironment()
    propose = (
        route_reference_proposal
        if args.planner == "route_reference"
        else scripted_proposal
    )
    report = dict(
        scope="control loop only; not teacher or SAC learner throughput",
        source_git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        scene_id=scenario.scenario_id,
        split=split,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        devices=[str(d) for d in jax.devices()],
        jax_enable_x64=jax.config.x64_enabled,
        provider=args.provider,
        planner=args.planner,
        requested_steps=args.steps,
        trials=[],
        status="RUNNING",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    try:
        for trial in range(args.trials):
            started = time.perf_counter()
            boundary = e.reset(scenario, seed=0, provider_spec=args.provider)
            jax.block_until_ready(e._state)
            entry = dict(trial=trial, reset_s=time.perf_counter() - started, steps=[])
            report["trials"].append(entry)
            for _ in range(args.steps):
                started = time.perf_counter()
                result, boundary = e.step_waypoints(
                    propose(e, boundary.token), boundary.token
                )
                jax.block_until_ready(e._state)
                entry["steps"].append(
                    dict(
                        tick=result.post_tick,
                        wall_s=time.perf_counter() - started,
                        source=result.source.value,
                        raw_check=result.check_result.reason,
                        terminal_reason=(
                            result.terminal_reason.value
                            if result.terminal_reason
                            else None
                        ),
                    )
                )
                if result.terminated or result.truncated:
                    break
            durations = [r["wall_s"] for r in entry["steps"]]
            entry.update(
                all_ticks=timing(durations),
                after_first_tick=timing(durations[1:]),
                execution_sources=dict(Counter(r["source"] for r in entry["steps"])),
            )
            write_json(args.output, report)
            print(
                json.dumps({k: v for k, v in entry.items() if k != "steps"}), flush=True
            )
        report["status"] = "COMPLETE"
    except BaseException as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
