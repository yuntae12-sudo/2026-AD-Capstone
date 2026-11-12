#!/usr/bin/env python3
"""Audit the frozen WOMD v2 target set without rescanning WOMD.

This script deliberately consumes only the existing 3,415-scenario locator and
the already extracted protobuf-topology / logged-trajectory evidence.  It does
not modify any v1/v2 artifact and does not label a topological convergence as
ON_RAMP when the available evidence cannot distinguish an entrance ramp from a
lane-drop/zipper geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "womd_gt_merge_audit_20261010"
MANUALLY_CONFIRMED_ON_RAMP: dict[str, str] = {}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_blank(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def event_classification(row: dict[str, Any]) -> dict[str, Any]:
    topology = row["topology_evidence"]
    interaction = row["interaction_evidence"]
    reason = row.get("automatic_reason") or ""
    front = interaction.get("front_vehicle_id")
    rear = interaction.get("rear_vehicle_id")

    source_history = int(interaction.get("source_occupancy_frames") or 0)
    target_stable = int(interaction.get("target_stable_frames") or 0)
    progress = float(interaction.get("longitudinal_progress_m") or 0.0)
    # These fields came from lane assignment/projection, not bbox-level physical
    # validation. Preserve the signal as a legacy transition candidate but do
    # not call it a completed merge or actual target entry.
    legacy_lane_transition = source_history >= 5 and target_stable >= 5 and progress >= 0.1
    physical_completion = False
    topological_convergence = bool(
        topology.get("source_exits_to_target")
        and len(set(topology.get("target_entry_lane_ids") or [])) >= 2
        and not topology.get("target_is_roundabout")
    )
    lane_change = bool(
        topology.get("is_lane_change_target")
        or reason in {"lane_change", "cut_in_not_topology_merge"}
    )

    if lane_change:
        road_type = "LANE_CHANGE"
        type_reason = "adjacent-lane topology or topology-free cut-in"
    elif topological_convergence:
        # Multiple predecessors prove convergence, but not the road's function.
        # Ramp-vs-lane-drop needs richer approach-lane context/manual map review.
        road_type = "UNCERTAIN"
        type_reason = "authoritative convergence; ON_RAMP vs ZIPPER unresolved"
    elif reason in {
        "serial_continuation",
        "diverge_split",
        "intersection_transition",
        "map_artifact",
        "no_authoritative_merge_type",
    }:
        road_type = "NON_MERGE"
        type_reason = reason
    else:
        road_type = "UNCERTAIN"
        type_reason = reason or "insufficient topology evidence"

    if row["candidate_id"] in MANUALLY_CONFIRMED_ON_RAMP and topological_convergence:
        road_type = "ON_RAMP"
        type_reason = MANUALLY_CONFIRMED_ON_RAMP[row["candidate_id"]]

    if reason == "target_not_stable" or (
        (topological_convergence or lane_change) and progress >= 0.1
    ):
        merge_state = "UNCERTAIN"
    elif road_type == "NON_MERGE" or reason == "no_physical_transition":
        merge_state = "NO_MERGE"
    else:
        merge_state = "UNCERTAIN"

    if front is not None and rear is not None:
        interaction_type = "UNVALIDATED_FRONT_REAR"
    elif front is not None:
        interaction_type = "UNVALIDATED_FRONT_ONLY"
    elif rear is not None:
        interaction_type = "UNVALIDATED_REAR_ONLY"
    else:
        interaction_type = "UNVALIDATED_NO_NEIGHBOR"

    full_observation = source_history >= 10 and target_stable >= 10
    if legacy_lane_transition and (topological_convergence or lane_change):
        suitability = "REVIEW"
        suitability_reason = "legacy lane transition; independent physical validation required"
    elif road_type == "UNCERTAIN":
        suitability = "REVIEW"
        suitability_reason = "topology or trajectory classification uncertain"
    else:
        suitability = "EXCLUDE"
        suitability_reason = "not a confirmed ON_RAMP merge"

    samples = row.get("gap_timeseries") or []
    completion = int(interaction.get("completion_frame") or 0)
    at_completion = min(samples, key=lambda x: abs(int(x["frame"]) - completion)) if samples else {}
    confidence = "HIGH" if road_type in {"ON_RAMP", "LANE_CHANGE", "NON_MERGE"} else "MEDIUM"

    return {
        "candidate_id": row["candidate_id"],
        "scenario_id": row["scenario_id"],
        "scene_key": row["scene_key"],
        "split": row["source_split"],
        "source_shard": row["source_shard"],
        "record_index": row["record_index"],
        "source_lane_id": topology["source_lane_id"],
        "target_lane_id": topology["target_lane_id"],
        "road_type": road_type,
        "road_type_reason": type_reason,
        "merge_state": merge_state,
        "front_rear_class": interaction_type,
        "front_vehicle_id": "" if front is None else front,
        "rear_vehicle_id": "" if rear is None else rear,
        "front_gap_m_at_completion": finite_or_blank(at_completion.get("front_gap_m", "")),
        "rear_gap_m_at_completion": finite_or_blank(at_completion.get("rear_gap_m", "")),
        "front_relative_speed_mps_at_completion": finite_or_blank(at_completion.get("front_relative_speed_mps", "")),
        "rear_relative_speed_mps_at_completion": finite_or_blank(at_completion.get("rear_relative_speed_mps", "")),
        "front_ttc_s_at_completion": finite_or_blank(at_completion.get("front_ttc_s", "")),
        "rear_ttc_s_at_completion": finite_or_blank(at_completion.get("rear_ttc_s", "")),
        "decision_start_frame": interaction["decision_start_frame"],
        "commit_frame": interaction["commit_frame"],
        "completion_frame": interaction["completion_frame"],
        "source_occupancy_frames": source_history,
        "target_stable_frames": target_stable,
        "valid_rollout_frames": source_history + target_stable,
        "longitudinal_progress_m": interaction["longitudinal_progress_m"],
        "lateral_displacement_m": interaction["lateral_displacement_m"],
        "target_entry_lane_ids": ";".join(map(str, topology.get("target_entry_lane_ids") or [])),
        "source_exit_lane_ids": ";".join(map(str, topology.get("source_exit_lane_ids") or [])),
        "actual_target_entry": False,
        "legacy_lane_id_transition": legacy_lane_transition,
        "physical_validation_status": "NOT_RUN",
        "stable_after_entry": target_stable >= 5,
        "full_approach_to_stabilization": full_observation,
        "classification_confidence": confidence,
        "suitability": suitability,
        "suitability_reason": suitability_reason,
        "v2_automatic_decision": row.get("automatic_decision") or "",
        "v2_automatic_reason": reason,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if not rows and not fields:
        raise ValueError(f"cannot infer CSV fields for empty output: {path}")
    fields = fields or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_event(rows: list[dict[str, Any]]) -> dict[str, Any]:
    road_rank = {"ON_RAMP": 5, "ZIPPER": 4, "UNCERTAIN": 3, "LANE_CHANGE": 2, "NON_MERGE": 1}
    state_rank = {"MERGE_COMPLETED": 4, "MERGE_ATTEMPTED": 3, "UNCERTAIN": 2, "NO_MERGE": 1}
    return max(
        rows,
        key=lambda x: (
            state_rank[x["merge_state"]],
            road_rank.get(x["road_type"], 0),
            x["target_stable_frames"],
            x["source_occupancy_frames"],
        ),
    )


def counter_rows(counter: Counter, *names: str) -> list[dict[str, Any]]:
    return [dict(zip(names, key if isinstance(key, tuple) else (key,)), count=value) for key, value in sorted(counter.items(), key=lambda x: str(x[0]))]


def numeric_summary(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    output = []
    for field in fields:
        values = sorted(
            float(row[field]) for row in rows
            if row.get(field) not in {None, ""} and math.isfinite(float(row[field]))
        )
        if not values:
            continue
        percentile = lambda p: values[round((len(values) - 1) * p)]
        output.append({
            "metric": field,
            "count": len(values),
            "p10": percentile(0.1),
            "median": percentile(0.5),
            "p90": percentile(0.9),
            "minimum": values[0],
            "maximum": values[-1],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    targets_path = ROOT / "data/manifests/v2/target_scenario_ids.csv"
    locator_path = ROOT / "data/manifests/v2/womd_scenario_proto_locator.csv"
    evidence_paths = [
        ROOT / "data/manifests/evidence_training.jsonl",
        ROOT / "data/manifests/evidence_validation.jsonl",
    ]
    canonical_path = ROOT / "data/manifests/merge_decision_manifest.csv"
    targets = read_csv(targets_path)
    locators = read_csv(locator_path)
    all_evidence = [row for path in evidence_paths for row in read_jsonl(path)]
    canonical = read_csv(canonical_path)

    target_ids = {row["scenario_id"] for row in targets}
    locator_by_id = {row["scenario_id"]: row for row in locators}
    if len(targets) != 3415 or len(target_ids) != 3415:
        raise RuntimeError(f"expected 3,415 unique targets, got {len(targets)} rows/{len(target_ids)} ids")
    if set(locator_by_id) != target_ids:
        raise RuntimeError("locator IDs do not exactly match frozen target IDs")
    evidence_ids = {row["scenario_id"] for row in all_evidence}
    if target_ids - evidence_ids:
        raise RuntimeError(f"evidence coverage mismatch: missing={len(target_ids-evidence_ids)}")
    # The evidence files predate the frozen target-ID list and retain 37
    # non-target scenarios.  Keep the audit population exactly equal to the
    # immutable locator population rather than silently expanding its scope.
    evidence = [row for row in all_evidence if row["scenario_id"] in target_ids]

    events = [event_classification(row) for row in evidence]
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_scenario[event["scenario_id"]].append(event)

    canonical_roles: dict[str, set[str]] = defaultdict(set)
    for row in canonical:
        canonical_roles[row["scenario_id"]].add(row["dataset_role"])

    scenario_rows = []
    for target in targets:
        scenario_id = target["scenario_id"]
        selected = best_event(by_scenario[scenario_id])
        locator = locator_by_id[scenario_id]
        roles = canonical_roles.get(scenario_id, set())
        old_status = "CORE" if "CORE" in roles else "SUPPORT" if "SUPPORT" in roles else "NOT_SELECTED"
        scenario_rows.append({
            "scenario_id": scenario_id,
            "split": target["source_split"],
            "source_shard": target["source_shard"],
            "candidate_count": target["candidate_count"],
            "proto_shard_index": locator["proto_shard_index"],
            "proto_record_index": locator["record_index"],
            "proto_source_object": locator["source_object"],
            "selected_candidate_id": selected["candidate_id"],
            "road_type": selected["road_type"],
            "merge_state": selected["merge_state"],
            "source_lane_id": selected["source_lane_id"],
            "target_lane_id": selected["target_lane_id"],
            "front_rear_class": selected["front_rear_class"],
            "front_vehicle_id": selected["front_vehicle_id"],
            "rear_vehicle_id": selected["rear_vehicle_id"],
            "actual_target_entry": selected["actual_target_entry"],
            "legacy_lane_id_transition": selected["legacy_lane_id_transition"],
            "physical_validation_status": selected["physical_validation_status"],
            "stable_after_entry": selected["stable_after_entry"],
            "valid_rollout_frames": selected["valid_rollout_frames"],
            "full_approach_to_stabilization": selected["full_approach_to_stabilization"],
            "classification_confidence": selected["classification_confidence"],
            "classification_reason": selected["road_type_reason"],
            "suitability": selected["suitability"],
            "suitability_reason": selected["suitability_reason"],
            "v2_membership": old_status,
        })

    write_csv(out / "maneuver_audit.csv", events)
    write_csv(out / "scenario_audit.csv", scenario_rows)

    stat_rows: list[dict[str, Any]] = []
    def add_counts(metric: str, values: Iterable[Any]) -> None:
        counts = Counter(values)
        total = sum(counts.values())
        for value, count in sorted(counts.items(), key=lambda x: str(x[0])):
            stat_rows.append({"metric": metric, "category": str(value), "count": count, "percent": round(100 * count / total, 3) if total else 0})

    add_counts("scenario_road_type", (x["road_type"] for x in scenario_rows))
    add_counts("scenario_merge_state", (x["merge_state"] for x in scenario_rows))
    add_counts("scenario_front_rear", (x["front_rear_class"] for x in scenario_rows))
    add_counts("scenario_suitability", (x["suitability"] for x in scenario_rows))
    add_counts("scenario_split", (x["split"] for x in scenario_rows))
    add_counts("maneuver_road_type", (x["road_type"] for x in events))
    add_counts("maneuver_merge_state", (x["merge_state"] for x in events))
    add_counts("maneuver_front_rear", (x["front_rear_class"] for x in events))
    write_csv(out / "statistics.csv", stat_rows)

    split_suitability = Counter((x["split"], x["suitability"]) for x in scenario_rows)
    cross = Counter((x["v2_membership"], x["road_type"], x["merge_state"]) for x in scenario_rows)
    write_csv(out / "split_suitability.csv", counter_rows(split_suitability, "split", "suitability"))
    write_csv(out / "v2_cross_tab.csv", counter_rows(cross, "v2_membership", "road_type", "merge_state"))
    completed_events = [row for row in events if row["merge_state"] == "MERGE_COMPLETED"]
    write_csv(out / "interaction_distribution.csv", numeric_summary(completed_events, [
        "front_gap_m_at_completion", "rear_gap_m_at_completion",
        "front_relative_speed_mps_at_completion", "rear_relative_speed_mps_at_completion",
        "front_ttc_s_at_completion", "rear_ttc_s_at_completion",
        "longitudinal_progress_m", "lateral_displacement_m",
    ]))

    split_ids: dict[str, set[str]] = defaultdict(set)
    for row in scenario_rows:
        split_ids[row["split"]].add(row["scenario_id"])
    overlap = sorted(split_ids["training"] & split_ids["validation"])
    completed = [x for x in scenario_rows if x["merge_state"] == "MERGE_COMPLETED"]
    full = [x for x in completed if x["full_approach_to_stabilization"]]
    legacy_front_rear = [
        x for x in scenario_rows
        if x["front_rear_class"] == "UNVALIDATED_FRONT_REAR"
        and x.get("legacy_lane_id_transition")
    ]
    summary = {
        "audit_kind": "WOMD_GROUND_TRUTH_DATASET_QUALITY_AUDIT_NOT_PPO_ROLLOUT",
        "target_scenarios": len(scenario_rows),
        "target_maneuvers": len(events),
        "ignored_non_target_evidence_rows": len(all_evidence) - len(evidence),
        "ignored_non_target_evidence_scenarios": len(evidence_ids - target_ids),
        "processed_success": len(scenario_rows),
        "processed_failure": 0,
        "uncertain_scenarios": sum(x["road_type"] == "UNCERTAIN" or x["merge_state"] == "UNCERTAIN" for x in scenario_rows),
        "confirmed_on_ramp": sum(x["road_type"] == "ON_RAMP" for x in scenario_rows),
        "confirmed_zipper": sum(x["road_type"] == "ZIPPER" for x in scenario_rows),
        "merge_completed": len(completed),
        "legacy_matcher_front_rear_lane_transitions": len(legacy_front_rear),
        "validated_front_rear_gap_merge_completed": 0,
        "interaction_validation_note": "Legacy front/rear IDs are not validated physical target-lane traffic; stationary and adjacent-lane false matches were confirmed during rollout review.",
        "full_approach_to_stabilization": len(full),
        "split_overlap_count": len(overlap),
        "split_overlap_scenario_ids": overlap,
        "local_proto_availability": {
            "training_target_scenarios": sum(x["split"] == "training" for x in scenario_rows),
            "training_local_shards": 0,
            "validation_target_scenarios": sum(x["split"] == "validation" for x in scenario_rows),
            "validation_local_shards": len(list((ROOT / "data/womd/scenario_proto/validation").glob("validation.tfrecord-*-of-00150"))),
        },
        "method_note": "Road type remains UNCERTAIN for authoritative multi-entry convergence because existing evidence does not prove ON_RAMP versus ZIPPER; completion is assessed independently.",
        "thresholds": {
            "min_source_occupancy_frames": 5,
            "min_target_stable_frames": 5,
            "min_longitudinal_progress_m": 0.1,
            "full_observation_source_frames": 10,
            "full_observation_target_frames": 10,
            "womd_dt_seconds": 0.1,
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [targets_path, locator_path, *evidence_paths, canonical_path]
        },
    }
    with (out / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
