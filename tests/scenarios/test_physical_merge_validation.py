import numpy as np

from src.scenarios.lane_geometry import LanePolyline, compute_arc_length
from src.scenarios.physical_merge_validation import (
    GapInterpretation,
    PhysicalMergeStatus,
    RoadContext,
    TrackRole,
    classify_physical_merge,
    classify_road_context,
    interpret_gap,
    validate_ego_lateral_entry,
    validate_target_lane_track,
    validated_ttc,
)


def lane(y=0.0):
    xy = np.asarray([[0.0, y], [10.0, y], [20.0, y]])
    direction = np.tile([1.0, 0.0], (3, 1))
    return LanePolyline(1, 1, xy, direction, compute_arc_length(xy))


def test_bbox_rejects_adjacent_lane_center_that_legacy_five_m_gate_accepted():
    xy = np.asarray([[i, 3.4] for i in range(8)], dtype=float)
    result = validate_target_lane_track(
        object_id=7, xy=xy, yaw=np.zeros(8), speed=np.ones(8) * 10,
        width=np.ones(8) * 2.0, valid=np.ones(8, bool), target_lane=lane(),
    )
    assert result.role is TrackRole.OUTSIDE_TARGET_LANE


def test_stationary_target_lane_vehicle_is_kept_but_not_moving_gap_boundary():
    xy = np.asarray([[4.0, 0.0]] * 8)
    result = validate_target_lane_track(
        object_id=8, xy=xy, yaw=np.zeros(8), speed=np.zeros(8),
        width=np.ones(8) * 1.8, valid=np.ones(8, bool), target_lane=lane(),
    )
    assert result.role is TrackRole.STATIONARY_TARGET_LANE


def test_moving_target_lane_track_requires_motion_and_persistence():
    xy = np.asarray([[i * 2.0, 0.0] for i in range(8)])
    result = validate_target_lane_track(
        object_id=9, xy=xy, yaw=np.zeros(8), speed=np.ones(8) * 20,
        width=np.ones(8) * 1.8, valid=np.ones(8, bool), target_lane=lane(),
    )
    assert result.role is TrackRole.MOVING_TARGET_LANE
    assert result.longest_persistence_frames == 8


def test_topology_is_not_an_input_to_on_ramp_context_confirmation():
    assert classify_road_context() is RoadContext.UNCERTAIN
    assert classify_road_context(
        source_is_feeder=True, target_is_through_mainline=True, is_lane_drop=False
    ) is RoadContext.ON_RAMP


def test_lane_id_change_without_lateral_entry_does_not_confirm_merge():
    xy = np.asarray([[i, 0.0] for i in range(10)], dtype=float)
    entry = validate_ego_lateral_entry(
        xy=xy, yaw=np.zeros(10), width=np.ones(10) * 1.8,
        valid=np.ones(10, bool), source_lane=lane(), target_lane=lane(), commit_frame=5,
    )
    decision = classify_physical_merge(road_context=RoadContext.ON_RAMP, ego_entry=entry)
    assert not entry.confirmed_lateral_entry
    assert decision.status is PhysicalMergeStatus.REJECTED


def test_negative_gap_and_ttc_zero_require_validated_same_lane_overlap():
    outside = validate_target_lane_track(
        object_id=7, xy=np.asarray([[i, 3.4] for i in range(8)]),
        yaw=np.zeros(8), speed=np.ones(8) * 10, width=np.ones(8) * 2,
        valid=np.ones(8, bool), target_lane=lane(),
    )
    artifact = interpret_gap(-1.0, True, outside)
    assert artifact is GapInterpretation.PROJECTION_ARTIFACT_OR_UNCERTAIN
    assert validated_ttc(-1.0, 5.0, artifact) is None

    stationary = validate_target_lane_track(
        object_id=8, xy=np.asarray([[4.0, 0.0]] * 8), yaw=np.zeros(8),
        speed=np.zeros(8), width=np.ones(8) * 1.8, valid=np.ones(8, bool),
        target_lane=lane(),
    )
    overlap = interpret_gap(-1.0, True, stationary)
    assert overlap is GapInterpretation.VALIDATED_OVERLAP
    assert validated_ttc(-1.0, 5.0, overlap) == 0.0
