"""Unbatched, full-length Scenario-proto loading for data inspection/replay.

No 91-frame slicing, object truncation, resampling, or future extrapolation.
This is a data adapter, not the merge training environment. The original proto
retains lane connectivity; Waymax roadgraph tensors contain geometry only.
SDC route generation and fixed-shape batching are separate future work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.scenarios.vendor.waymo_open_dataset.protos import scenario_pb2


DEFAULT_ROOT = Path("/mnt/e/Datasets/WOMD/v_1_3_1/scenario/training_20s")
DEFAULT_OUTPUT = Path("/mnt/e/Datasets/WOMD/v_1_3_1/derived/training_20s")
TRAJECTORY_FIELDS = {
    "x": "center_x", "y": "center_y", "z": "center_z",
    "vel_x": "velocity_x", "vel_y": "velocity_y", "yaw": "heading",
    "length": "length", "width": "width", "height": "height",
}


def raw_records(path):
    """TensorFlow validates TFRecord framing/CRCs while iterating."""
    import tensorflow as tf

    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = tf.data.TFRecordDataset(str(path)).with_options(options)
    for raw in dataset:
        yield bytes(raw.numpy())


def parse_scenario(raw):
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(raw)
    return scenario


def read_scenario(path, record_index=0):
    if record_index < 0:
        raise ValueError("record_index must be nonnegative")
    for index, raw in enumerate(raw_records(path)):
        if index == record_index:
            return parse_scenario(raw)
    raise IndexError(f"Record {record_index} not found in {path}")


def validate_scenario(scenario):
    """Check basic structure before indexing or allocating Waymax arrays."""
    times = np.asarray(scenario.timestamps_seconds, dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Expected at least two finite, strictly increasing timestamps")
    if not scenario.scenario_id:
        raise ValueError("Missing scenario_id")
    if not 0 <= scenario.sdc_track_index < len(scenario.tracks):
        raise ValueError("Invalid sdc_track_index")
    if not 0 <= scenario.current_time_index < len(times):
        raise ValueError("Invalid current_time_index")
    if any(len(track.states) != len(times) for track in scenario.tracks):
        raise ValueError("Track state length differs from timestamp count")
    if len({track.id for track in scenario.tracks}) != len(scenario.tracks):
        raise ValueError("Duplicate track IDs")
    if len({feature.id for feature in scenario.map_features}) != len(scenario.map_features):
        raise ValueError("Duplicate map feature IDs")
    # No dynamic states is permitted; partial time alignment is ambiguous.
    if len(scenario.dynamic_map_states) not in (0, len(times)):
        raise ValueError("Dynamic map state length differs from timestamp count")
    if any(not 0 <= p.track_index < len(scenario.tracks) for p in scenario.tracks_to_predict):
        raise ValueError("Invalid tracks_to_predict index")
    for dynamic in scenario.dynamic_map_states:
        ids = [light.lane for light in dynamic.lane_states]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate traffic-light lane in a frame")
    return times


def checked_int32(values):
    data = np.asarray(values, dtype=np.int64)
    bounds = np.iinfo(np.int32)
    if np.any(data < bounds.min) or np.any(data > bounds.max):
        raise ValueError("ID/time exceeds Waymax int32 range")
    return data.astype(np.int32)


def roadgraph_arrays(scenario):
    points, directions, types, ids = [], [], [], []
    counts = {}
    for feature in scenario.map_features:
        kind = feature.WhichOneof("feature_data")
        counts[kind or "unset"] = counts.get(kind or "unset", 0) + 1
        if kind in ("lane", "road_line", "road_edge"):
            geometry = getattr(feature, kind)
            xyz = [(p.x, p.y, p.z) for p in geometry.polyline]
            type_id = int(geometry.type) + {"lane": 0, "road_line": 5, "road_edge": 14}[kind]
        elif kind == "stop_sign":
            p = feature.stop_sign.position
            xyz, type_id = [(p.x, p.y, p.z)], 17
        elif kind in ("crosswalk", "speed_bump", "driveway"):
            xyz = [(p.x, p.y, p.z) for p in getattr(feature, kind).polygon]
            type_id = {"crosswalk": 18, "speed_bump": 19, "driveway": -1}[kind]
            if xyz and xyz[0] != xyz[-1]:
                xyz.append(xyz[0])
        else:
            raise ValueError(f"Unsupported/unset map feature {feature.id}: {kind}")
        if not xyz:
            continue
        xyz = np.asarray(xyz, dtype=np.float32)
        if not np.isfinite(xyz).all():
            raise ValueError(f"Nonfinite map coordinates: {feature.id}")
        delta = np.zeros_like(xyz)
        if len(xyz) > 1:
            delta[:-1] = np.diff(xyz, axis=0)
            delta[-1] = delta[-2]
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        direction = np.divide(delta, norms, out=np.zeros_like(delta), where=norms > 0)
        points.extend(xyz)
        directions.extend(direction)
        types.extend([type_id] * len(xyz))
        ids.extend([feature.id] * len(xyz))
    # One invalid placeholder makes no-map/no-light cases compatible with viewers.
    n = len(points)
    xyz = np.asarray(points or [(0, 0, 0)], dtype=np.float32)
    direction = np.asarray(directions or [(0, 0, 0)], dtype=np.float32)
    result = dict(zip(("x", "y", "z"), xyz.T))
    result.update(zip(("dir_x", "dir_y", "dir_z"), direction.T))
    result.update(types=np.asarray(types or [-1], dtype=np.int32),
                  ids=checked_int32(ids or [-1]), valid=np.full(max(1, n), n > 0))
    return result, counts


def traffic_light_arrays(scenario, num_frames):
    lanes = sorted({light.lane for frame in scenario.dynamic_map_states for light in frame.lane_states})
    rows = {lane: i for i, lane in enumerate(lanes)}
    shape = (max(1, len(lanes)), num_frames)
    result = {name: np.zeros(shape, np.float32) for name in ("x", "y", "z")}
    result.update(state=np.zeros(shape, np.int32), lane_ids=np.full(shape, -1, np.int32),
                  valid=np.zeros(shape, bool))
    for lane in lanes:
        result["lane_ids"][rows[lane], :] = checked_int32(lane)
    for t, frame in enumerate(scenario.dynamic_map_states):
        for light in frame.lane_states:
            row = rows[light.lane]
            for axis in ("x", "y", "z"):
                result[axis][row, t] = getattr(light.stop_point, axis)
            result["state"][row, t] = light.state
            result["valid"][row, t] = True
    if not all(np.isfinite(result[a]).all() for a in ("x", "y", "z")):
        raise ValueError("Nonfinite traffic light coordinates")
    return result


def scenario_to_waymax(scenario):
    """Return (SimulatorState, conversion report), retaining every track/frame.

    Time is rebased to the first sample to avoid absolute-epoch int32 overflow.
    Source timestamps and topology remain in the caller's original Scenario.
    Simulation is initialized at frame zero; future sim frames remain invalid.
    """
    import jax
    import jax.numpy as jnp
    from waymax import datatypes

    times = validate_scenario(scenario)
    valid = np.asarray([[state.valid for state in track.states] for track in scenario.tracks])
    arrays = {}
    for target, source in TRAJECTORY_FIELDS.items():
        value = np.asarray([[getattr(state, source) for state in track.states]
                            for track in scenario.tracks], dtype=np.float32)
        if not np.isfinite(value[valid]).all():
            raise ValueError(f"Nonfinite valid track values: {source}")
        arrays[target] = np.where(valid, value, 0).astype(np.float32)
    timestamps = checked_int32(np.rint((times - times[0]) * 1e6))
    arrays.update(valid=valid, timestamp_micros=np.broadcast_to(timestamps, valid.shape).copy())
    log = datatypes.Trajectory(**{k: jnp.asarray(v) for k, v in arrays.items()})
    sim = jax.tree_util.tree_map(lambda v: jnp.zeros_like(v).at[:, 0].set(v[:, 0]), log)
    num_objects = len(scenario.tracks)
    modeled = np.zeros(num_objects, bool)
    for prediction in scenario.tracks_to_predict:
        modeled[prediction.track_index] = True
    metadata = datatypes.ObjectMetadata(
        ids=jnp.asarray(checked_int32([track.id for track in scenario.tracks])),
        object_types=jnp.asarray([track.object_type for track in scenario.tracks], dtype=jnp.int32),
        is_valid=jnp.ones(num_objects, dtype=bool), is_modeled=jnp.asarray(modeled),
        is_sdc=jnp.arange(num_objects) == scenario.sdc_track_index,
        objects_of_interest=jnp.asarray([track.id in scenario.objects_of_interest for track in scenario.tracks]),
        is_controlled=jnp.zeros(num_objects, dtype=bool),
    )
    road, counts = roadgraph_arrays(scenario)
    lights = traffic_light_arrays(scenario, len(times))
    roadgraph = datatypes.RoadgraphPoints(**{k: jnp.asarray(v) for k, v in road.items()})
    traffic = datatypes.TrafficLights(**{k: jnp.asarray(v) for k, v in lights.items()})
    state = datatypes.SimulatorState(
        sim_trajectory=sim, log_trajectory=log, object_metadata=metadata,
        log_traffic_light=traffic, roadgraph_points=roadgraph,
        timestep=jnp.asarray(0, dtype=jnp.int32), sdc_paths=None,
    )
    for component in (log, sim, metadata, roadgraph, traffic, state):
        component.validate()
    report = {
        "scenario_id": scenario.scenario_id, "frames": len(times),
        "duration_s": float(times[-1] - times[0]), "timestamp_origin_s": float(times[0]),
        "source_current_time_index": scenario.current_time_index,
        "tracks": num_objects, "sdc_track_index": scenario.sdc_track_index,
        "sdc_valid_frames": int(valid[scenario.sdc_track_index].sum()),
        "roadgraph_points": int(road["valid"].sum()), "map_feature_counts": counts,
        "traffic_light_lanes": int(np.any(lights["valid"], axis=1).sum()),
        "traffic_light_valid_samples": int(lights["valid"].sum()),
        "sdc_paths_loaded": False,
        "notes": ["Unbatched log-replay adapter; not a PPO/SAC training environment.",
                  "Original proto retains topology. No route/SDC paths are reconstructed.",
                  "Driveway geometry uses UNKNOWN (-1) in installed Waymax's map enum.",
                  "Per-frame dimensions are preserved; invalid track values are zero-masked.",
                  "No trajectory padding, object truncation, or time resampling."],
    }
    return state, report
