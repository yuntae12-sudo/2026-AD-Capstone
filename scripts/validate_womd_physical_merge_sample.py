#!/usr/bin/env python3
"""Validate a fixed 16-scenario WOMD sample without generating success GIFs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_womd_gt_merge_rollouts import lane_xy, load_scenario, track_by_object_id
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length
from src.scenarios.physical_merge_validation import (
    RoadContext,
    classify_physical_merge,
    validate_ego_lateral_entry,
    validate_target_lane_track,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/womd_gt_merge_audit_20261010"
OUTPUT = AUDIT / "physical_validation_sample"
REGRESSION_SCENARIOS = {"a650cfdc29e7cc3f", "c6201759ab89810d"}


def _polyline(scenario, lane_id):
    xy = lane_xy(scenario, lane_id)
    segment = np.diff(xy, axis=0)
    norm = np.maximum(np.linalg.norm(segment, axis=1, keepdims=True), 1e-9)
    direction = np.vstack([segment / norm, segment[-1:] / norm[-1:]])
    return LanePolyline(int(lane_id), 1, xy, direction, compute_arc_length(xy))


def _arrays(track, frame_window=None):
    states = track.states
    arrays = {
        "xy": np.asarray([(s.center_x, s.center_y) for s in states]),
        "yaw": np.asarray([s.heading for s in states]),
        "speed": np.asarray([np.hypot(s.velocity_x, s.velocity_y) for s in states]),
        "length": np.asarray([s.length for s in states]),
        "width": np.asarray([s.width for s in states]),
        "valid": np.asarray([s.valid for s in states]),
    }
    if frame_window is not None:
        start, end = frame_window
        window_mask = np.zeros(len(states), dtype=bool)
        window_mask[max(0, start):min(len(states), end + 1)] = True
        arrays["valid"] &= window_mask
    return arrays


def _metadata_paths():
    paths = sorted((AUDIT / "review_candidates").glob("*/metadata.json"))
    paths += sorted((AUDIT / "corrected_review").glob("*/metadata.json"))
    paths += [next((AUDIT / "rejected_rollouts_stationary_neighbors").glob("*57__t17__112_122/metadata.json"))]
    return paths


def main():
    old_by_id = {
        row["scenario_id"]: row
        for row in csv.DictReader((AUDIT / "scenario_audit.csv").open(encoding="utf-8"))
    }
    rows = []
    for metadata_path in _metadata_paths():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scenario, _, _ = load_scenario(metadata["scenario_id"])
        source = _polyline(scenario, metadata["source_lane_id"])
        target = _polyline(scenario, metadata["target_lane_id"])
        ego = scenario.tracks[scenario.sdc_track_index]
        ego_arrays = _arrays(ego)
        entry = validate_ego_lateral_entry(
            xy=ego_arrays["xy"], yaw=ego_arrays["yaw"], width=ego_arrays["width"],
            length=ego_arrays["length"], valid=ego_arrays["valid"],
            source_lane=source, target_lane=target,
            commit_frame=int(metadata["keyframes"]["during"]),
        )

        neighbor_results = {}
        neighbor_window = (
            int(metadata["keyframes"]["before"]),
            int(metadata["keyframes"]["completed"]),
        )
        for name in ("front", "rear"):
            object_id = metadata.get(f"{name}_vehicle_id")
            if object_id is None:
                neighbor_results[name] = None
                continue
            track = track_by_object_id(scenario, object_id)
            neighbor_results[name] = validate_target_lane_track(
                object_id=int(object_id), target_lane=target,
                **_arrays(track, neighbor_window)
            )

        # Road function cannot be inferred from entry_lanes/exit_lanes alone.
        # No independent feeder/mainline annotation exists in this 16-case
        # material, so every road-context result remains explicitly uncertain.
        decision = classify_physical_merge(
            road_context=RoadContext.UNCERTAIN, ego_entry=entry,
            front=neighbor_results["front"], rear=neighbor_results["rear"],
        )
        old = old_by_id[metadata["scenario_id"]]
        rows.append({
            "scenario_id": metadata["scenario_id"],
            "candidate_id": metadata["candidate_id"],
            "regression_case": metadata["scenario_id"] in REGRESSION_SCENARIOS,
            "legacy_merge_state": old["merge_state"],
            "legacy_front_rear_class": old["front_rear_class"],
            "road_context": decision.road_context.value,
            "ego_lateral_entry": entry.confirmed_lateral_entry,
            "ego_entry_reason": entry.reason,
            "source_pre_occupancy_fraction": round(entry.source_pre_occupancy_fraction, 3),
            "target_pre_outside_fraction": round(entry.target_pre_outside_fraction, 3),
            "target_post_occupancy_fraction": round(entry.target_post_occupancy_fraction, 3),
            "lateral_approach_m": round(entry.lateral_approach_m, 3),
            "front_id": metadata.get("front_vehicle_id"),
            "front_role": neighbor_results["front"].role.value if neighbor_results["front"] else "NONE",
            "front_path_m": round(neighbor_results["front"].path_length_m, 3) if neighbor_results["front"] else "",
            "front_speed_mps": round(neighbor_results["front"].median_speed_mps, 3) if neighbor_results["front"] else "",
            "front_offset_m": round(neighbor_results["front"].median_center_offset_m, 3) if neighbor_results["front"] else "",
            "rear_id": metadata.get("rear_vehicle_id"),
            "rear_role": neighbor_results["rear"].role.value if neighbor_results["rear"] else "NONE",
            "rear_path_m": round(neighbor_results["rear"].path_length_m, 3) if neighbor_results["rear"] else "",
            "rear_speed_mps": round(neighbor_results["rear"].median_speed_mps, 3) if neighbor_results["rear"] else "",
            "rear_offset_m": round(neighbor_results["rear"].median_center_offset_m, 3) if neighbor_results["rear"] else "",
            "revised_judgment": decision.status.value,
            "revised_reason": decision.reason,
            "review_image": str(metadata_path.parent.joinpath("overview_highres.png").relative_to(ROOT)),
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "manual_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (OUTPUT / "manual_review.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Physical merge manual-review index", "",
        "16개 고정 validation 사례다. `revised_judgment`가 `UNCERTAIN`인 사례를 성공으로 사용하지 않는다.",
        "정지 차량도 삭제하지 않고 `STATIONARY_TARGET_LANE` 또는 차로 밖 객체로 기록한다. GIF는 생성하지 않았다.", "",
        "| scenario | legacy | ego lateral entry | front / rear | revised | still |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        marker = " (regression)" if row["regression_case"] else ""
        image = Path(row["review_image"])
        relative_image = Path("..") / image.relative_to("outputs/womd_gt_merge_audit_20261010")
        lines.append(
            f"| `{row['scenario_id']}`{marker} | {row['legacy_merge_state']} / {row['legacy_front_rear_class']} "
            f"| {row['ego_lateral_entry']} | {row['front_role']} / {row['rear_role']} "
            f"| {row['revised_judgment']} | [PNG]({relative_image.as_posix()}) |"
        )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
