#!/usr/bin/env python3
"""Inventory development splits and verify real reset eligibility, not teacher quality.

Stage screen uses the production route builder with source values rounded exactly
as in the Waymax adapter. Stage verify applies the full, unchanged environment
reset to cached candidates. No policy is run on the held-out test split.
"""

import argparse
from collections import Counter
import gc
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
from src.environment.womd_control.control import Ego
from src.environment.womd_control.geometry import ResetRejected, build_route
from src.scenarios.womd20 import parse_scenario, raw_records, validate_scenario


def development_split(scenario_id):
    if not scenario_id:
        raise ValueError("Missing scenario/group ID")
    bucket = (
        int(hashlib.sha256(("split_v2|" + scenario_id).encode()).hexdigest(), 16) % 100
    )
    return ("train" if bucket < 80 else "validation" if bucket < 90 else "test"), bucket


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prefilter(scenario):
    """Cheap necessary checks only. This never certifies full reset eligibility."""
    times = validate_scenario(scenario)
    if np.any(np.abs(np.diff(times) - 0.1) > 0.005):
        raise ResetRejected("SOURCE_TIMESTEP_OUTSIDE_TOLERANCE")
    states = scenario.tracks[scenario.sdc_track_index].states
    starts = [
        i
        for i in range(max(2, scenario.current_time_index), len(times) - 50)
        if all(states[j].valid for j in range(i - 2, i + 1))
    ]
    if not starts:
        raise ResetRejected("INSUFFICIENT_HISTORY_OR_REMAINING_TIME")
    frame = starts[0]
    if not all(s.valid for s in states[frame:]):
        raise ResetRejected("EGO_LOG_VALIDITY_GAP")
    s, previous = states[frame], states[frame - 1]

    def converted(values):
        return np.asarray(values, np.float32).astype(np.float64)

    velocity = converted([s.velocity_x, s.velocity_y])
    ego = Ego(
        converted([s.center_x, s.center_y]),
        velocity,
        (velocity - converted([previous.velocity_x, previous.velocity_y])) / 0.1,
        True,
        float(np.float32(s.heading)),
        float(np.float32(s.length)),
        float(np.float32(s.width)),
    )
    if not (
        0 <= ego.speed <= 15
        and min(ego.length, ego.width) > 0
        and np.linalg.norm(ego.acceleration) <= 6
    ):
        raise ResetRejected("INITIAL_STATE_OUTSIDE_ODD")
    route = build_route(scenario, ego, frame)
    return dict(
        start_frame=frame,
        last_frame=len(times) - 1,
        remaining_ticks_upper_bound=min(200, len(times) - 1 - frame),
        initial_speed_mps=ego.speed,
        route_lane_ids=route.lane_ids,
        boundary_feature_ids=route.boundary_ids,
    )


def screen(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "candidates"
    cache.mkdir(exist_ok=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    version = hashlib.sha256(
        Path(__file__).read_bytes()
        + (ROOT / "src/environment/womd_control/geometry.py").read_bytes()
    ).hexdigest()
    for shard_index in range(args.start_shard, args.start_shard + args.shards):
        path = args.root / f"training_20s.tfrecord-{shard_index:05d}-of-01000"
        destination = args.output / f"screen-{shard_index:05d}.json"
        stat = path.stat()
        fingerprint = dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if destination.exists():
            old = json.loads(destination.read_text())
            if (
                old["source_fingerprint"] != fingerprint
                or old["screen_code_sha256"] != version
            ):
                raise ValueError("Source/code changed; use a new output directory")
            print(f"resume shard={shard_index}", flush=True)
            continue
        started = time.perf_counter()
        rows = []
        for index, raw in enumerate(raw_records(path)):
            row = dict(
                source_shard=path.name,
                record_index=index,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
            )
            try:
                scenario = parse_scenario(raw)
                split, bucket = development_split(scenario.scenario_id)
                row.update(
                    scene_id=scenario.scenario_id,
                    group_id=scenario.scenario_id,
                    group_status="scenario_fallback_overlap_unverified",
                    split=split,
                    bucket=bucket,
                )
                if split == "test":
                    row.update(status="HELD_OUT", reason="test_not_screened")
                else:
                    row.update(prefilter(scenario), status="ROUTE_CANDIDATE")
                    name = f"{shard_index:05d}-{index:05d}.pb"
                    (cache / name).write_bytes(raw)
                    row["cached_proto"] = "candidates/" + name
            except ResetRejected as error:
                row.update(status="PREFILTER_REJECTED", reason=str(error))
            except (ValueError, IndexError) as error:
                row.update(
                    status="DATA_ERROR", reason=f"{type(error).__name__}: {error}"
                )
            rows.append(row)
        report = dict(
            source_shard=path.name,
            source_fingerprint=fingerprint,
            source_git_head=head,
            screen_code_sha256=version,
            scope="development split; route prefilter only; test untouched",
            elapsed_s=time.perf_counter() - started,
            rows=rows,
        )
        write_json(destination, report)
        print(
            f"shard={shard_index} records={len(rows)} statuses={dict(Counter(r['status'] for r in rows))} seconds={report['elapsed_s']:.1f}",
            flush=True,
        )
    summarize(args.output)


def verify(args):
    import jax
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.contracts.planning import ContractError

    source_files = (
        sorted((ROOT / "src/contracts").glob("*.py"))
        + sorted((ROOT / "src/environment/womd_control").glob("*.py"))
        + [ROOT / "src/scenarios/womd20.py", Path(__file__)]
    )
    runtime_digest = hashlib.sha256(
        b"".join(p.read_bytes() for p in source_files)
    ).hexdigest()

    candidates = [
        r
        for p in sorted(args.output.glob("screen-*.json"))
        for r in json.loads(p.read_text())["rows"]
        if r["status"] == "ROUTE_CANDIDATE"
    ]
    for row in candidates:
        name = Path(row["cached_proto"]).stem
        destination = args.output / f"reset-{name}.json"
        raw = (args.output / row["cached_proto"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["raw_sha256"]:
            raise ValueError("Candidate checksum mismatch")
        if destination.exists():
            old = json.loads(destination.read_text())
            if (
                old["raw_sha256"] != row["raw_sha256"]
                or old.get("runtime_code_sha256") != runtime_digest
            ):
                raise ValueError(
                    "Candidate/runtime changed; use a new output directory"
                )
            continue
        results = []
        for provider in ("fixed_keep", "fixed_follow"):
            started = time.perf_counter()
            e = SACPlanningEnvironment()
            result = dict(provider=provider)
            try:
                boundary = e.reset(parse_scenario(raw), seed=0, provider_spec=provider)
                result.update(
                    status="RESET_ACCEPTED",
                    start_frame=e.start_frame,
                    remaining_ticks_upper_bound=min(200, e.last_frame - e.start_frame),
                    target_agent_id=e.command.draft.target_agent_id,
                    observation_features=boundary.sac_observation.flatten().size,
                    route_lane_ids=e.route.lane_ids,
                    boundary_feature_ids=e.route.boundary_ids,
                )
            except ResetRejected as error:
                result.update(status="RESET_REJECTED", reason=str(error))
            except (ContractError, ValueError) as error:
                result.update(
                    status="RESET_ERROR", reason=f"{type(error).__name__}: {error}"
                )
            result["elapsed_s"] = time.perf_counter() - started
            results.append(result)
            del e
            jax.clear_caches()
            gc.collect()
        write_json(
            destination,
            dict(row, reset_results=results, runtime_code_sha256=runtime_digest),
        )
        print(
            f"reset {name} split={row['split']} {[(r['provider'],r['status'],r.get('reason')) for r in results]}",
            flush=True,
        )
        summarize(args.output)
    summarize(args.output)


def summarize(output):
    reports = [json.loads(p.read_text()) for p in sorted(output.glob("screen-*.json"))]
    rows = [r for p in reports for r in p["rows"]]
    ids = Counter(r.get("scene_id") for r in rows if r.get("scene_id"))
    duplicates = sorted(i for i, n in ids.items() if n > 1)
    # Duplicate scene IDs are quarantined, never silently counted twice.
    resets = [json.loads(p.read_text()) for p in sorted(output.glob("reset-*.json"))]
    manifest = []
    for row in resets:
        if row["scene_id"] in duplicates:
            continue
        for result in row["reset_results"]:
            if result["status"] == "RESET_ACCEPTED":
                manifest.append(
                    {k: v for k, v in row.items() if k != "reset_results"} | result
                )
    write_json(
        output / "eligible_manifest.json",
        dict(
            split_version="split_v2",
            group_status="scenario_fallback_overlap_unverified",
            scope="reset-eligible only; not teacher-certified or independent test",
            rows=manifest,
        ),
    )
    write_json(
        output / "summary.json",
        dict(
            scanned_shards=len(reports),
            scanned_records=len(rows),
            scan_wall_s=sum(p["elapsed_s"] for p in reports),
            splits=dict(Counter(r.get("split", "unknown") for r in rows)),
            statuses=dict(Counter(r["status"] for r in rows)),
            rejection_reasons=dict(
                Counter(
                    r.get("reason", "").split(":")[0]
                    for r in rows
                    if r["status"] == "PREFILTER_REJECTED"
                )
            ),
            duplicate_scene_ids=duplicates,
            reset_verified_records=len(resets),
            reset_reasons=dict(
                Counter(
                    r.get("reason", "ACCEPTED")
                    for row in resets
                    for r in row["reset_results"]
                )
            ),
            eligible_by_split_provider=dict(
                Counter(f"{r['split']}:{r['provider']}" for r in manifest)
            ),
            upper_bound_ticks_by_split_provider={
                key: sum(
                    r["remaining_ticks_upper_bound"]
                    for r in manifest
                    if f"{r['split']}:{r['provider']}" == key
                )
                for key in sorted({f"{r['split']}:{r['provider']}" for r in manifest})
            },
            full_dataset_scan=False,
            teacher_transitions_collected=0,
            independent_groups_verified=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "verify", "summarize"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=10)
    args = parser.parse_args()
    if (
        not 0 <= args.start_shard < 1000
        or not 1 <= args.shards <= 1000 - args.start_shard
    ):
        parser.error("shard range must be within 0..999")
    if args.command == "screen" and args.root is None:
        parser.error("screen requires --root")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    if args.command == "summarize":
        summarize(args.output)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
