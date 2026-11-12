"""Conservative, offline physical validation for WOMD merge candidates.

This module deliberately sits after the legacy single-frame interaction
matcher.  A topology edge, a lane-id transition, or a projected longitudinal
gap is evidence to review a candidate; none is sufficient to confirm a merge.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional

import numpy as np

from src.scenarios.lane_geometry import LanePolyline, project_point_to_polyline


class RoadContext(enum.Enum):
    ON_RAMP = "ON_RAMP"
    ZIPPER = "ZIPPER"
    LANE_CHANGE = "LANE_CHANGE"
    NON_MERGE = "NON_MERGE"
    UNCERTAIN = "UNCERTAIN"


class TrackRole(enum.Enum):
    MOVING_TARGET_LANE = "MOVING_TARGET_LANE"
    STATIONARY_TARGET_LANE = "STATIONARY_TARGET_LANE"
    OUTSIDE_TARGET_LANE = "OUTSIDE_TARGET_LANE"
    HEADING_MISMATCH = "HEADING_MISMATCH"
    INSUFFICIENT_PERSISTENCE = "INSUFFICIENT_PERSISTENCE"
    UNCERTAIN = "UNCERTAIN"


class PhysicalMergeStatus(enum.Enum):
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"


class GapInterpretation(enum.Enum):
    SEPARATED = "SEPARATED"
    VALIDATED_OVERLAP = "VALIDATED_OVERLAP"
    PROJECTION_ARTIFACT_OR_UNCERTAIN = "PROJECTION_ARTIFACT_OR_UNCERTAIN"


@dataclasses.dataclass(frozen=True)
class PhysicalValidationConfig:
    # WOMD does not expose lane polygons here.  A 3.7 m nominal lane with a
    # 0.25 m boundary band is a conservative proxy; the vehicle half-width is
    # included, so a centre point in an adjacent lane cannot pass.
    nominal_lane_width_m: float = 3.7
    boundary_tolerance_m: float = 0.25
    max_heading_difference_deg: float = 15.0
    min_persistence_frames: int = 5
    min_strict_occupancy_fraction: float = 0.8
    moving_min_path_length_m: float = 2.0
    moving_min_median_speed_mps: float = 1.0
    stationary_max_path_length_m: float = 1.0
    stationary_max_median_speed_mps: float = 0.5
    min_ego_lateral_entry_m: float = 1.0
    min_target_stable_frames: int = 5


@dataclasses.dataclass(frozen=True)
class TrackValidation:
    object_id: int
    role: TrackRole
    valid_frames: int
    longest_persistence_frames: int
    strict_occupancy_fraction: float
    boundary_fraction: float
    median_center_offset_m: float
    median_heading_difference_deg: float
    path_length_m: float
    median_speed_mps: float


@dataclasses.dataclass(frozen=True)
class EgoEntryValidation:
    confirmed_lateral_entry: bool
    source_pre_occupancy_fraction: float
    target_pre_outside_fraction: float
    target_post_occupancy_fraction: float
    target_post_stable_frames: int
    lateral_approach_m: float
    reason: str


@dataclasses.dataclass(frozen=True)
class PhysicalMergeDecision:
    status: PhysicalMergeStatus
    road_context: RoadContext
    ego_entry: EgoEntryValidation
    front: Optional[TrackValidation]
    rear: Optional[TrackValidation]
    reason: str

    @property
    def moving_front_id(self) -> Optional[int]:
        return self.front.object_id if self.front and self.front.role is TrackRole.MOVING_TARGET_LANE else None

    @property
    def moving_rear_id(self) -> Optional[int]:
        return self.rear.object_id if self.rear and self.rear.role is TrackRole.MOVING_TARGET_LANE else None


def _wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _lane_metrics(xy, yaw, width, valid, lane: LanePolyline, config, length=None):
    strict = np.zeros(len(valid), dtype=bool)
    boundary = np.zeros(len(valid), dtype=bool)
    offsets = np.full(len(valid), np.nan)
    heading_diff = np.full(len(valid), np.nan)
    half_lane = config.nominal_lane_width_m / 2.0
    length = np.zeros(len(valid), dtype=float) if length is None else np.asarray(length, dtype=float)
    for i in np.flatnonzero(valid):
        projection = project_point_to_polyline(lane, float(xy[i, 0]), float(xy[i, 1]))
        offset = abs(float(projection["lateral_distance_m"]))
        offsets[i] = offset
        heading_diff[i] = abs(_wrap_deg(np.degrees(float(yaw[i]) - projection["heading_rad"])))
        angle = np.radians(heading_diff[i])
        # Lateral support of the oriented rectangle, not merely its centre.
        bbox_half_extent = (
            abs(np.sin(angle)) * max(0.0, float(length[i])) / 2.0
            + abs(np.cos(angle)) * max(0.0, float(width[i])) / 2.0
        )
        footprint_extent = offset + bbox_half_extent
        strict[i] = footprint_extent <= half_lane - config.boundary_tolerance_m
        boundary[i] = (
            half_lane - config.boundary_tolerance_m < footprint_extent
            <= half_lane + config.boundary_tolerance_m
        )
    return strict, boundary, offsets, heading_diff


def validate_target_lane_track(
    *, object_id: int, xy: np.ndarray, yaw: np.ndarray, speed: np.ndarray,
    width: np.ndarray, valid: np.ndarray, target_lane: LanePolyline,
    length: Optional[np.ndarray] = None,
    config: PhysicalValidationConfig = PhysicalValidationConfig(),
) -> TrackValidation:
    """Validate bbox occupancy, heading, temporal identity, and motion."""

    xy = np.asarray(xy, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    speed = np.asarray(speed, dtype=float)
    width = np.asarray(width, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    strict, boundary, offsets, heading = _lane_metrics(
        xy, yaw, width, valid, target_lane, config, length
    )
    n = int(valid.sum())
    persistence = _longest_true_run(valid)
    occupancy = float(strict.sum() / n) if n else 0.0
    boundary_fraction = float(boundary.sum() / n) if n else 0.0
    median_offset = float(np.nanmedian(offsets)) if n else float("nan")
    median_heading = float(np.nanmedian(heading)) if n else float("nan")
    valid_xy = xy[valid]
    path = float(np.hypot(*np.diff(valid_xy, axis=0).T).sum()) if len(valid_xy) > 1 else 0.0
    median_speed = float(np.nanmedian(np.abs(speed[valid]))) if n else 0.0

    if persistence < config.min_persistence_frames:
        role = TrackRole.INSUFFICIENT_PERSISTENCE
    elif median_heading > config.max_heading_difference_deg:
        role = TrackRole.HEADING_MISMATCH
    elif occupancy < config.min_strict_occupancy_fraction:
        role = TrackRole.OUTSIDE_TARGET_LANE if boundary_fraction < 0.5 else TrackRole.UNCERTAIN
    elif path <= config.stationary_max_path_length_m and median_speed <= config.stationary_max_median_speed_mps:
        role = TrackRole.STATIONARY_TARGET_LANE
    elif path >= config.moving_min_path_length_m and median_speed >= config.moving_min_median_speed_mps:
        role = TrackRole.MOVING_TARGET_LANE
    else:
        role = TrackRole.UNCERTAIN

    return TrackValidation(
        object_id=object_id, role=role, valid_frames=n,
        longest_persistence_frames=persistence,
        strict_occupancy_fraction=occupancy, boundary_fraction=boundary_fraction,
        median_center_offset_m=median_offset,
        median_heading_difference_deg=median_heading,
        path_length_m=path, median_speed_mps=median_speed,
    )


def validate_ego_lateral_entry(
    *, xy: np.ndarray, yaw: np.ndarray, width: np.ndarray, valid: np.ndarray,
    source_lane: LanePolyline, target_lane: LanePolyline, commit_frame: int,
    length: Optional[np.ndarray] = None,
    config: PhysicalValidationConfig = PhysicalValidationConfig(),
) -> EgoEntryValidation:
    """Require source occupancy followed by bbox occupancy of the target lane."""

    valid = np.asarray(valid, dtype=bool)
    pre_slice = np.arange(max(0, commit_frame - config.min_target_stable_frames), commit_frame)
    post_slice = np.arange(commit_frame, min(len(valid), commit_frame + config.min_target_stable_frames))
    source_strict, _, _, _ = _lane_metrics(xy, yaw, width, valid, source_lane, config, length)
    target_strict, _, target_offsets, _ = _lane_metrics(xy, yaw, width, valid, target_lane, config, length)

    def fraction(mask, indices):
        usable = indices[valid[indices]]
        return float(mask[usable].mean()) if len(usable) else 0.0

    source_pre = fraction(source_strict, pre_slice)
    target_pre_outside = 1.0 - fraction(target_strict, pre_slice)
    target_post = fraction(target_strict, post_slice)
    stable = _longest_true_run(target_strict[post_slice] & valid[post_slice])
    pre_offsets = target_offsets[pre_slice][valid[pre_slice]]
    post_offsets = target_offsets[post_slice][valid[post_slice]]
    approach = (
        float(np.nanmedian(pre_offsets) - np.nanmedian(post_offsets))
        if len(pre_offsets) and len(post_offsets) else 0.0
    )
    confirmed = (
        source_pre >= 0.8 and target_pre_outside >= 0.8 and target_post >= 0.8
        and stable >= config.min_target_stable_frames
        and approach >= config.min_ego_lateral_entry_m
    )
    failed = []
    if source_pre < 0.8: failed.append("source_pre_occupancy")
    if target_pre_outside < 0.8: failed.append("already_in_target_or_ambiguous")
    if target_post < 0.8: failed.append("target_bbox_not_occupied")
    if stable < config.min_target_stable_frames: failed.append("target_not_stable")
    if approach < config.min_ego_lateral_entry_m: failed.append("no_lateral_entry")
    return EgoEntryValidation(
        confirmed_lateral_entry=confirmed,
        source_pre_occupancy_fraction=source_pre,
        target_pre_outside_fraction=target_pre_outside,
        target_post_occupancy_fraction=target_post,
        target_post_stable_frames=stable,
        lateral_approach_m=approach,
        reason="confirmed" if confirmed else ";".join(failed),
    )


def classify_road_context(
    *, source_is_feeder: Optional[bool] = None,
    target_is_through_mainline: Optional[bool] = None,
    is_lane_drop: Optional[bool] = None,
    is_intersection: Optional[bool] = None,
    is_lane_change: Optional[bool] = None,
) -> RoadContext:
    """Classify road function; map connectivity alone is intentionally absent."""

    if is_intersection is True:
        return RoadContext.NON_MERGE
    if is_lane_change is True:
        return RoadContext.LANE_CHANGE
    if source_is_feeder is True and target_is_through_mainline is True:
        return RoadContext.ZIPPER if is_lane_drop is True else RoadContext.ON_RAMP if is_lane_drop is False else RoadContext.UNCERTAIN
    return RoadContext.UNCERTAIN


def classify_physical_merge(
    *, road_context: RoadContext, ego_entry: EgoEntryValidation,
    front: Optional[TrackValidation] = None, rear: Optional[TrackValidation] = None,
) -> PhysicalMergeDecision:
    if road_context in {RoadContext.NON_MERGE, RoadContext.LANE_CHANGE}:
        return PhysicalMergeDecision(PhysicalMergeStatus.REJECTED, road_context, ego_entry, front, rear, "road_context_reject")
    if road_context is not RoadContext.ON_RAMP:
        return PhysicalMergeDecision(PhysicalMergeStatus.UNCERTAIN, road_context, ego_entry, front, rear, "on_ramp_not_confirmed")
    if not ego_entry.confirmed_lateral_entry:
        return PhysicalMergeDecision(PhysicalMergeStatus.REJECTED, road_context, ego_entry, front, rear, ego_entry.reason)
    return PhysicalMergeDecision(PhysicalMergeStatus.CONFIRMED, road_context, ego_entry, front, rear, "physical_on_ramp_entry_confirmed")


def interpret_gap(gap_m: float, ego_target_occupied: bool, other: Optional[TrackValidation]) -> GapInterpretation:
    """A negative 1-D projected gap is overlap only after lane validation."""

    if gap_m > 0.0:
        return GapInterpretation.SEPARATED
    if ego_target_occupied and other and other.role in {TrackRole.MOVING_TARGET_LANE, TrackRole.STATIONARY_TARGET_LANE}:
        return GapInterpretation.VALIDATED_OVERLAP
    return GapInterpretation.PROJECTION_ARTIFACT_OR_UNCERTAIN


def validated_ttc(gap_m: float, closing_speed_mps: float, interpretation: GapInterpretation) -> Optional[float]:
    """Return None for an invalid projection; reserve zero for validated overlap."""

    if interpretation is GapInterpretation.PROJECTION_ARTIFACT_OR_UNCERTAIN:
        return None
    if interpretation is GapInterpretation.VALIDATED_OVERLAP:
        return 0.0
    if closing_speed_mps <= 0.0:
        return float("inf")
    return float(gap_m / closing_speed_mps)


__all__ = [
    "RoadContext", "TrackRole", "PhysicalMergeStatus", "GapInterpretation",
    "PhysicalValidationConfig", "TrackValidation", "EgoEntryValidation",
    "PhysicalMergeDecision", "validate_target_lane_track",
    "validate_ego_lateral_entry", "classify_road_context",
    "classify_physical_merge", "interpret_gap", "validated_ttc",
]
