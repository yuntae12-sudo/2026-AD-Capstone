#!/usr/bin/env python3
"""Render logged WOMD Scenario trajectories (never PPO rollouts).

The renderer uses the frozen Scenario-protobuf locator for direct access to a
small number of already-local validation records. It creates still review
material; a GIF is permitted only for a candidate explicitly marked
``CONFIRMED_ON_RAMP`` after physical validation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scenarios.scenario_proto_loader import iter_scenario_protobufs


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "outputs/womd_gt_merge_audit_20261010"
DEFAULT_CANDIDATES = (
    "validation_tfexample.tfrecord-00003-of-00150#57__t17__112_122",  # front+rear
    "validation_tfexample.tfrecord-00002-of-00150#139__t54__142_202",  # front only
    "validation_tfexample.tfrecord-00003-of-00150#65__t51__65_143",  # no neighbor
)
MANUAL_CLASSIFICATIONS: dict[str, tuple[str, str]] = {}


def load_evidence() -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("evidence_training.jsonl", "evidence_validation.jsonl"):
        with (ROOT / "data/manifests" / name).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                result[row["candidate_id"]] = row
    return result


def load_scenario(scenario_id: str):
    locators = {
        row["scenario_id"]: row
        for row in csv.DictReader((ROOT / "data/manifests/v2/womd_scenario_proto_locator.csv").open())
    }
    locator = locators[scenario_id]
    if locator["proto_split"] != "validation":
        raise RuntimeError(f"local GT rendering is validation-only; got {locator['proto_split']}")
    shard = ROOT / "data/womd/scenario_proto/validation" / (
        f"validation.tfrecord-{int(locator['proto_shard_index']):05d}-of-00150"
    )
    if not shard.exists():
        raise FileNotFoundError(shard)
    wanted = int(locator["record_index"])
    for index, scenario in enumerate(iter_scenario_protobufs([str(shard)])):
        if index == wanted:
            if scenario.scenario_id != scenario_id:
                raise RuntimeError(f"locator drift: expected {scenario_id}, got {scenario.scenario_id}")
            return scenario, locator, shard
    raise RuntimeError(f"record {wanted} not found in {shard}")


def lane_xy(scenario, lane_id: int) -> np.ndarray:
    for feature in scenario.map_features:
        if int(feature.id) == lane_id and feature.HasField("lane"):
            return np.asarray([(p.x, p.y) for p in feature.lane.polyline], dtype=float)
    raise KeyError(f"lane {lane_id} not found")


def track_by_object_id(scenario, object_id: int | None):
    if object_id is None:
        return None
    return next((track for track in scenario.tracks if int(track.id) == int(object_id)), None)


def track_arrays(track) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray([(s.center_x, s.center_y) for s in track.states], dtype=float)
    heading = np.asarray([s.heading for s in track.states], dtype=float)
    valid = np.asarray([s.valid for s in track.states], dtype=bool)
    return xy, heading, valid


def bounds_for(source: np.ndarray, target: np.ndarray, ego_xy: np.ndarray, ego_valid: np.ndarray):
    pts = np.vstack([source, target, ego_xy[ego_valid]])
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    center = (lo + hi) / 2
    span = max(float(np.max(hi - lo)) + 20.0, 70.0)
    span = min(span, 180.0)
    return center[0] - span / 2, center[0] + span / 2, center[1] - span / 2, center[1] + span / 2


def draw_vehicle(ax, xy, heading, color, label=None, size=55, marker="o"):
    ax.scatter([xy[0]], [xy[1]], s=size, c=color, marker=marker, edgecolors="black", linewidths=0.5, zorder=8, label=label)
    ax.arrow(xy[0], xy[1], 2.2 * math.cos(heading), 2.2 * math.sin(heading), color=color, width=0.12, zorder=9)


def render_frame(
    scenario,
    evidence,
    frame: int,
    output_path: Path | None,
    *,
    dpi: int,
    return_image: bool = False,
):
    topology = evidence["topology_evidence"]
    interaction = evidence["interaction_evidence"]
    source_id = int(topology["source_lane_id"])
    target_id = int(topology["target_lane_id"])
    source = lane_xy(scenario, source_id)
    target = lane_xy(scenario, target_id)
    ego = scenario.tracks[scenario.sdc_track_index]
    ego_xy, ego_heading, ego_valid = track_arrays(ego)
    front_id = interaction.get("front_vehicle_id")
    rear_id = interaction.get("rear_vehicle_id")
    front = track_by_object_id(scenario, front_id)
    rear = track_by_object_id(scenario, rear_id)
    xmin, xmax, ymin, ymax = bounds_for(source, target, ego_xy, ego_valid)

    fig, ax = plt.subplots(figsize=(9, 9))
    for feature in scenario.map_features:
        if not feature.HasField("lane") or int(feature.id) in {source_id, target_id}:
            continue
        xy = np.asarray([(p.x, p.y) for p in feature.lane.polyline], dtype=float)
        if len(xy) > 1 and np.any((xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)):
            ax.plot(xy[:, 0], xy[:, 1], color="#c6c6c6", linewidth=0.8, alpha=0.65, zorder=1)
    ax.plot(source[:, 0], source[:, 1], color="#1677ff", linewidth=4, label=f"Source lane {source_id}", zorder=3)
    ax.plot(target[:, 0], target[:, 1], color="#ff8c1a", linewidth=4, label=f"Target lane {target_id}", zorder=3)
    ax.plot(ego_xy[ego_valid, 0], ego_xy[ego_valid, 1], color="#d7191c", linewidth=1.5, alpha=0.25, label="Ego GT full path")
    past = ego_valid.copy()
    past[frame + 1 :] = False
    ax.plot(ego_xy[past, 0], ego_xy[past, 1], color="#d7191c", linewidth=3, zorder=5)

    for track in scenario.tracks:
        if track is ego or frame >= len(track.states) or not track.states[frame].valid:
            continue
        if front is not None and track.id == front.id:
            continue
        if rear is not None and track.id == rear.id:
            continue
        state = track.states[frame]
        if xmin <= state.center_x <= xmax and ymin <= state.center_y <= ymax:
            ax.scatter(state.center_x, state.center_y, s=14, c="#888888", alpha=0.55, zorder=4)

    if ego_valid[frame]:
        draw_vehicle(ax, ego_xy[frame], ego_heading[frame], "#d7191c", "Ego (WOMD GT)", 95, "s")
    for track, color, label in ((front, "#00a65a", f"Front {front_id}"), (rear, "#8e44ad", f"Rear {rear_id}")):
        if track is not None and frame < len(track.states) and track.states[frame].valid:
            state = track.states[frame]
            draw_vehicle(ax, (state.center_x, state.center_y), state.heading, color, label, 75, "D")

    commit = int(interaction["commit_frame"])
    completion = int(interaction["completion_frame"])
    if frame < commit:
        phase = "APPROACH / GAP_INTERACTION"
    elif frame == commit:
        phase = "MERGE_EXECUTION"
    else:
        phase = "MERGE_COMPLETION / STABILIZATION"
    ax.set_title(
        "WOMD GROUND TRUTH (NOT PPO)\n"
        f"{scenario.scenario_id} | frame {frame:02d} ({frame * 0.1:.1f}s) | {phase}",
        fontsize=12,
    )
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(alpha=0.15); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig); buf.seek(0)
        image = Image.open(buf).convert("RGB")
        image.load()
        return image
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_candidate(candidate_id: str, evidence: dict[str, Any], output_root: Path, *, still_only: bool = False) -> None:
    scenario, locator, shard = load_scenario(evidence["scenario_id"])
    safe = candidate_id.replace("#", "__").replace("/", "_")
    out = output_root / safe
    out.mkdir(parents=True, exist_ok=True)
    interaction = evidence["interaction_evidence"]
    commit = int(interaction["commit_frame"])
    completion = int(interaction["completion_frame"])
    valid_ego = [i for i, state in enumerate(scenario.tracks[scenario.sdc_track_index].states) if state.valid]
    start, end = min(valid_ego), max(valid_ego)
    before = max(start, commit - max(5, min(20, commit - start)))
    during = commit
    after = min(end, completion + max(5, min(20, end - completion)))
    keyframes = {"before": before, "during": during, "completed": after}
    for name, frame in keyframes.items():
        render_frame(scenario, evidence, frame, out / f"{name}.png", dpi=220)
    render_frame(scenario, evidence, after, out / "overview_highres.png", dpi=300)

    frame_step = 2
    if not still_only:
        frames = [render_frame(scenario, evidence, frame, None, dpi=80, return_image=True) for frame in range(start, end + 1, frame_step)]
        frames[0].save(
            out / "womd_gt_rollout.gif",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
            optimize=False,
        )
    topology = evidence["topology_evidence"]
    classification, classification_note = MANUAL_CLASSIFICATIONS.get(candidate_id, (
        "UNCERTAIN",
        "Legacy lane-id transition only; physical lateral entry, road function, and target-lane traffic are not confirmed.",
    ))
    metadata = {
        "artifact_kind": "WOMD_GROUND_TRUTH_ROLLOUT_NOT_PPO",
        "candidate_id": candidate_id,
        "scenario_id": scenario.scenario_id,
        "ego_id": int(scenario.tracks[scenario.sdc_track_index].id),
        "front_vehicle_id": interaction.get("front_vehicle_id"),
        "rear_vehicle_id": interaction.get("rear_vehicle_id"),
        "source_lane_id": topology["source_lane_id"],
        "target_lane_id": topology["target_lane_id"],
        "source_exit_lane_ids": topology["source_exit_lane_ids"],
        "target_entry_lane_ids": topology["target_entry_lane_ids"],
        "classification": classification,
        "classification_note": classification_note,
        "physical_merge_status": classification,
        "legacy_lane_id_merge_state": "MERGE_COMPLETED",
        "legacy_completion_basis": {
            "source_occupancy_frames": interaction["source_occupancy_frames"],
            "target_stable_frames": interaction["target_stable_frames"],
            "longitudinal_progress_m": interaction["longitudinal_progress_m"],
            "lateral_displacement_m": interaction["lateral_displacement_m"],
        },
        "keyframes": keyframes,
        "source_proto_shard": str(shard.relative_to(ROOT)),
        "source_proto_record_index": int(locator["record_index"]),
        "simulation_dt_seconds": 0.1,
        "gif_frame_stride": frame_step,
        "gif_frame_duration_seconds": 0.2,
        "playback_relation": "1.0x real time (0.2 s represented per GIF frame and 0.2 s displayed)",
        "partial_gt": False,
        "gif_valid_frame_range": [start, end],
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_ids", nargs="*", default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR / "gt_rollouts")
    parser.add_argument("--still-only", action="store_true")
    args = parser.parse_args()
    evidence_by_id = load_evidence()
    for candidate_id in args.candidate_ids:
        manual = MANUAL_CLASSIFICATIONS.get(candidate_id)
        if not args.still_only and (manual is None or manual[0] != "CONFIRMED_ON_RAMP"):
            raise RuntimeError(
                "refusing unverified success GIF; run --still-only until "
                f"physical ON_RAMP validation is confirmed: {candidate_id}"
            )
        evidence = evidence_by_id[candidate_id]
        topology = evidence["topology_evidence"]
        interaction = evidence["interaction_evidence"]
        if not (topology["source_exits_to_target"] and len(set(topology["target_entry_lane_ids"])) >= 2):
            raise RuntimeError(f"not an authoritative multi-entry convergence: {candidate_id}")
        if interaction["target_stable_frames"] < 5:
            raise RuntimeError(f"not a completed/stable GT transition: {candidate_id}")
        render_candidate(candidate_id, evidence, args.output_dir, still_only=args.still_only)


if __name__ == "__main__":
    main()
