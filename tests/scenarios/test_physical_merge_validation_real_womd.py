"""Small local-data regressions for the two audited false positives."""

from pathlib import Path

import numpy as np
import pytest

from scripts.render_womd_gt_merge_rollouts import lane_xy, load_scenario, track_by_object_id
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length
from src.scenarios.physical_merge_validation import TrackRole, validate_target_lane_track


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "a650cfdc29e7cc3f": {"target_lane_id": 122, "object_ids": (667, 664)},
    "c6201759ab89810d": {"target_lane_id": 311, "object_ids": (211,)},
}


def _polyline(scenario, lane_id):
    xy = lane_xy(scenario, lane_id)
    segment = np.diff(xy, axis=0)
    norm = np.maximum(np.linalg.norm(segment, axis=1, keepdims=True), 1e-9)
    direction = np.vstack([segment / norm, segment[-1:] / norm[-1:]])
    return LanePolyline(lane_id, 1, xy, direction, compute_arc_length(xy))


def _track_arrays(track):
    states = track.states
    return {
        "xy": np.asarray([(s.center_x, s.center_y) for s in states]),
        "yaw": np.asarray([s.heading for s in states]),
        "speed": np.asarray([np.hypot(s.velocity_x, s.velocity_y) for s in states]),
        "length": np.asarray([s.length for s in states]),
        "width": np.asarray([s.width for s in states]),
        "valid": np.asarray([s.valid for s in states]),
    }


@pytest.mark.parametrize("scenario_id", tuple(CASES))
def test_known_false_positive_is_not_a_moving_target_lane_gap(scenario_id):
    locator = ROOT / "data/manifests/v2/womd_scenario_proto_locator.csv"
    if not locator.exists():
        pytest.skip("local WOMD locator is unavailable")
    try:
        scenario, _, _ = load_scenario(scenario_id)
    except FileNotFoundError:
        pytest.skip("required local WOMD validation shard is unavailable")

    case = CASES[scenario_id]
    target = _polyline(scenario, case["target_lane_id"])
    results = []
    for object_id in case["object_ids"]:
        track = track_by_object_id(scenario, object_id)
        assert track is not None
        results.append(validate_target_lane_track(
            object_id=object_id, target_lane=target, **_track_arrays(track)
        ))

    assert all(result.role is not TrackRole.MOVING_TARGET_LANE for result in results)
    # These IDs are retained in the regression rather than silently deleted:
    # the validator must explain whether they are stationary or outside lane.
    assert all(result.role in {TrackRole.STATIONARY_TARGET_LANE, TrackRole.OUTSIDE_TARGET_LANE, TrackRole.UNCERTAIN} for result in results)
