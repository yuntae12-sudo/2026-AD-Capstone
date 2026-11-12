"""Full-length loading, time alignment, and replay boundary regression tests."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
import tensorflow as tf

from scripts.prepare_womd20 import replay
from src.scenarios.vendor.waymo_open_dataset.protos import scenario_pb2
from src.scenarios.womd20 import scenario_to_waymax, validate_scenario
from src.scenarios.womd20_index import scan_index


def make_scenario(frames=198, scenario_id="synthetic_20s"):
    scenario = scenario_pb2.Scenario(scenario_id=scenario_id, sdc_track_index=1, current_time_index=10)
    scenario.timestamps_seconds.extend(1_700_000_000 + np.arange(frames) * 0.1)
    for i in range(2):
        track = scenario.tracks.add(id=100 + i, object_type=1)
        for t in range(frames):
            track.states.add(center_x=i * 5 + t, center_y=i, center_z=0,
                             length=4.5, width=2, height=1.5, heading=0.1,
                             velocity_x=10, velocity_y=0, valid=(i == 1 or 3 <= t < frames - 3))
    scenario.tracks_to_predict.add(track_index=1)
    scenario.objects_of_interest.append(100)
    lane = scenario.map_features.add(id=7).lane
    lane.type = 2
    lane.polyline.add(x=0, y=0, z=0)
    lane.polyline.add(x=200, y=0, z=0)
    for t in range(frames):
        dynamic = scenario.dynamic_map_states.add()
        if t != 2:
            light = dynamic.lane_states.add(lane=7, state=4 if t < 100 else 6)
            light.stop_point.x = 100
        if t > 90:
            light = dynamic.lane_states.add(lane=8, state=5)
            light.stop_point.x = 120
    return scenario


def test_preserves_full_time_axis_all_objects_and_light_identity():
    scenario = make_scenario()
    state, report = scenario_to_waymax(scenario)
    assert state.log_trajectory.shape == (2, 198)
    assert state.log_traffic_light.shape == (2, 198)
    assert int(state.remaining_timesteps) == 197
    assert not bool(state.is_done)
    np.testing.assert_array_equal(np.asarray(state.object_metadata.is_sdc), [False, True])
    np.testing.assert_array_equal(np.asarray(state.object_metadata.is_modeled), [False, True])
    assert not np.asarray(state.sim_trajectory.valid)[:, 1:].any()
    assert not bool(state.log_trajectory.valid[0, 0])
    assert bool(state.log_trajectory.valid[0, 3])
    # Absolute epoch micros exceed int32. Relative micros must remain correct.
    assert int(state.log_trajectory.timestamp_micros[1, 197]) == 19_700_000
    assert report["timestamp_origin_s"] == 1_700_000_000
    assert not bool(state.log_traffic_light.valid[0, 2])
    assert not bool(state.log_traffic_light.valid[1, 90])
    assert bool(state.log_traffic_light.valid[1, 91])
    assert int(state.log_traffic_light.lane_ids[1, 197]) == 8
    assert int(state.log_traffic_light.state[0, 99]) == 4
    assert int(state.log_traffic_light.state[0, 100]) == 6
    assert int(state.log_trajectory.x[1, 197]) == 202
    assert state.sdc_paths is None


def test_replay_goes_past_nine_seconds_and_stops_at_actual_end(tmp_path):
    scenario = make_scenario()
    state, _ = scenario_to_waymax(scenario)
    result = replay(scenario, state, tmp_path, render=False)
    assert result["checked_frames"] == 198
    assert result["frames_after_9s"] > 100
    assert result["final_frame"] == 197
    assert result["final_done"]
    assert result["remaining_timesteps"] == 0
    assert result["all_logged_fields_match"]


def test_no_traffic_lights_and_driveway_geometry_are_explicit():
    scenario = make_scenario()
    del scenario.dynamic_map_states[:]
    driveway = scenario.map_features.add(id=20).driveway
    driveway.polygon.add(x=1, y=1, z=0)
    driveway.polygon.add(x=2, y=1, z=0)
    driveway.polygon.add(x=1, y=2, z=0)
    state, report = scenario_to_waymax(scenario)
    assert not np.asarray(state.log_traffic_light.valid).any()
    assert report["traffic_light_lanes"] == 0
    road = state.roadgraph_points
    mask = np.asarray(road.ids) == 20
    assert mask.sum() == 4  # Closed polygon retains all vertices.
    assert (np.asarray(road.types)[mask] == -1).all()


@pytest.mark.parametrize("fault", ["timestamps", "track_length", "signals", "duplicate_track", "sdc"])
def test_rejects_misaligned_or_ambiguous_source_data(fault):
    scenario = make_scenario()
    if fault == "timestamps":
        scenario.timestamps_seconds[1] = scenario.timestamps_seconds[0]
    elif fault == "track_length":
        del scenario.tracks[0].states[-1]
    elif fault == "signals":
        del scenario.dynamic_map_states[-1]
    elif fault == "duplicate_track":
        scenario.tracks[1].id = scenario.tracks[0].id
    else:
        scenario.sdc_track_index = 2
    with pytest.raises(ValueError):
        validate_scenario(scenario)


def write_records(path, records):
    with tf.io.TFRecordWriter(str(path)) as writer:
        for record in records:
            writer.write(record)


def test_index_resumes_completed_shard_and_rechecks_changed_file(tmp_path, monkeypatch):
    root, output = tmp_path / "source", tmp_path / "output"
    root.mkdir()
    path = root / "training_20s.tfrecord-00000-of-01000"
    write_records(path, [make_scenario().SerializeToString()])
    first = scan_index(root, output, limit_shards=1)
    assert first["scenarios"] == 1 and first["failed_shards"] == 0
    assert not first["full_scan_ok"]  # Pilot is never a full dataset pass.
    import src.scenarios.womd20_index as module
    original = module.raw_records
    monkeypatch.setattr(module, "raw_records", lambda _: pytest.fail("Should resume without reading"))
    scan_index(root, output, limit_shards=1)
    monkeypatch.setattr(module, "raw_records", original)
    write_records(path, [make_scenario().SerializeToString(), b"\xff"])
    second = scan_index(root, output, limit_shards=1)
    assert second["scenarios"] == 2
    assert second["invalid_records"] == 1
    assert second["failed_shards"] == 1
    # Repaired shard replaces its previous rows rather than duplicating them.
    write_records(path, [make_scenario(scenario_id="repaired").SerializeToString()])
    third = scan_index(root, output, limit_shards=1)
    assert third["scenarios"] == 1 and third["failed_shards"] == 0


def test_index_reports_crc_corruption_and_rejects_different_root(tmp_path):
    root, output = tmp_path / "source", tmp_path / "output"
    root.mkdir()
    path = root / "training_20s.tfrecord-00000-of-01000"
    write_records(path, [make_scenario().SerializeToString()])
    payload = bytearray(path.read_bytes())
    payload[20] ^= 1
    path.write_bytes(payload)
    result = scan_index(root, output, limit_shards=1)
    assert result["failed_shards"] == 1
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="root mismatch"):
        scan_index(other, output, limit_shards=1)
