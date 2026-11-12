"""Current/past-only feature assembly for the development fixed provider."""

import numpy as np

from src.contracts.observation import CORE_SHAPES, ITEM_SHAPES, CoreObservation
from src.contracts.planning import DT, Direction, Maneuver, Source, Stage, rotation
from .geometry import arc, project, sample, wrap


def build_observation(environment):
    e = environment
    ego, route, command = e.ego, e.route, e.command
    r = rotation(ego.yaw)
    features = {k: np.zeros(shape, np.float32) for k, shape in CORE_SHAPES.items()}
    masks = {k: np.zeros(shape, bool) for k, shape in CORE_SHAPES.items()}
    items = {k: np.zeros(shape, bool) for k, shape in ITEM_SHAPES.items()}
    clipped = 0

    def write(name, values, valid=None, scale=1.0):
        nonlocal clipped
        values = np.asarray(values, np.float64) / np.asarray(scale)
        valid = np.ones_like(values, bool) if valid is None else np.asarray(valid, bool)
        if not np.isfinite(values[valid]).all():
            raise ValueError(f"nonfinite observed {name}")
        clipped += int(np.count_nonzero(valid & (np.abs(values) > 5)))
        features[name][...] = np.where(valid, np.clip(values, -5, 5), 0).astype(
            np.float32
        )
        masks[name][...] = valid

    s, projected, tangent = project(route.center, ego.position)
    lateral = (
        -tangent[1] * (ego.position - projected)[0]
        + tangent[0] * (ego.position - projected)[1]
    )
    heading_error = wrap(ego.yaw - np.arctan2(tangent[1], tangent[0]))
    previous = e.ego_at(e.frame - 1)
    local_v, local_a = ego.velocity @ r, ego.acceleration @ r
    write("task", [1, 0, float(e.goal_satisfied)])
    write(
        "ego",
        [
            *local_v,
            ego.speed,
            wrap(ego.yaw - previous.yaw) / DT,
            *local_a,
            ego.length,
            ego.width,
            lateral,
            np.sin(heading_error),
            np.cos(heading_error),
            max(0.0, route.goal_s - s),
            e.tick * DT,
            (e.last_frame - e.frame) * DT,
        ],
        [True] * 4 + [ego.acceleration_valid] * 2 + [True] * 8,
        [30, 30, 30, 1, 6, 6, 10, 10, 100, 1, 1, 100, 20, 20],
    )
    trajectory = e._state.sim_trajectory
    ids = np.asarray(e._state.object_metadata.ids)
    current_valid = np.asarray(trajectory.valid[:, e.frame])
    current_xy = np.stack(
        [np.asarray(trajectory.x[:, e.frame]), np.asarray(trajectory.y[:, e.frame])], -1
    )
    distances = np.linalg.norm(current_xy - ego.position, axis=1)
    required = [
        x
        for x in (
            command.draft.target_agent_id,
            command.draft.preferred_front_id,
            command.draft.preferred_rear_id,
        )
        if x is not None
    ]
    indices = [i for i in range(len(ids)) if current_valid[i] and i != e.ego_index]
    front, rear = [], []
    closest_approach = {}
    for i in indices:
        position = current_xy[i]
        velocity = np.array(
            [trajectory.vel_x[i, e.frame], trajectory.vel_y[i, e.frame]], float
        )
        relative_p, relative_v = position - ego.position, velocity - ego.velocity
        closest_time = np.clip(
            -np.dot(relative_p, relative_v)
            / max(np.dot(relative_v, relative_v), 1e-12),
            0,
            3,
        )
        closest_approach[i] = float(
            np.linalg.norm(relative_p + relative_v * closest_time)
        )
        if route.contains(
            position,
            float(trajectory.yaw[i, e.frame]),
            float(trajectory.length[i, e.frame]),
            float(trajectory.width[i, e.frame]),
        ):
            offset = project(route.center, position)[0] - s
            (front if offset >= 0 else rear).append((abs(offset), int(ids[i])))
    near_ids = {min(group)[1] for group in (front, rear) if group}
    indices.sort(
        key=lambda i: (
            0 if int(ids[i]) in required else 1,
            0 if int(ids[i]) in near_ids else 1,
            closest_approach[i],
            float(distances[i]),
            int(ids[i]),
        )
    )
    indices = indices[:32]
    agent_values = np.zeros((3, 32, 17))
    agent_valid = np.zeros_like(agent_values, bool)
    for slot, index in enumerate(indices):
        for history, frame in enumerate(range(e.frame - 2, e.frame + 1)):
            if not bool(trajectory.valid[index, frame]):
                continue
            p = np.array(
                [trajectory.x[index, frame], trajectory.y[index, frame]], float
            )
            v = np.array(
                [trajectory.vel_x[index, frame], trajectory.vel_y[index, frame]], float
            )
            past_ego = e.ego_at(frame)
            yaw = float(trajectory.yaw[index, frame]) - ego.yaw
            kind = int(e._state.object_metadata.object_types[index])
            type_features = [float(kind == k) for k in (1, 2, 3)] + [
                float(kind not in (1, 2, 3))
            ]
            agent_values[history, slot] = [
                *((p - ego.position) @ r),
                *((v - past_ego.velocity) @ r),
                np.sin(yaw),
                np.cos(yaw),
                float(trajectory.length[index, frame]),
                float(trajectory.width[index, frame]),
                *type_features,
                float(int(ids[index]) == command.draft.target_agent_id),
                float(int(ids[index]) == command.draft.preferred_front_id),
                float(int(ids[index]) == command.draft.preferred_rear_id),
                0.0,
                0.0,
            ]
            agent_valid[history, slot] = True
            items["agents"][history, slot] = True
    # If the fixed FOLLOW target is temporarily absent, reserve a current slot
    # and distinguish the estimate from measured history.
    if e.target_estimated and e.target_cache is not None:
        free = next((i for i in range(32) if not items["agents"][2, i]), None)
        if free is None:
            free = 31
        agent_values[:, free], agent_valid[:, free], items["agents"][:, free] = (
            0,
            False,
            False,
        )
        target = e.target_cache
        age = e.tick - e.target_last_tick
        position = target.position + target.velocity * (age * DT)
        yaw = target.yaw - ego.yaw
        agent_values[2, free] = [
            *((position - ego.position) @ r),
            *((target.velocity - ego.velocity) @ r),
            np.sin(yaw),
            np.cos(yaw),
            target.length,
            target.width,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            age * DT,
        ]
        agent_valid[2, free], items["agents"][2, free] = True, True
    write(
        "agents",
        agent_values,
        agent_valid,
        [100, 100, 30, 30, 1, 1, 10, 10] + [1] * 8 + [20],
    )
    distances = s + np.arange(40) * 2.0
    points = sample(route.center, distances)
    directions = np.array([project(route.center, p)[2] for p in points])
    route_values = np.concatenate([(points - ego.position) @ r, directions @ r], axis=1)
    route_valid = np.repeat((distances <= route.length)[:, None], 4, axis=1)
    items["route"] = route_valid[:, 0]
    write("route", route_values, route_valid, [100, 100, 1, 1])
    lane_values = np.zeros((16, 20, 7))
    lane_valid = np.zeros_like(lane_values, bool)
    features_by_id = {int(f.id): f for f in e._scenario.map_features}
    lane_candidates = []
    for lane_id, feature in features_by_id.items():
        if feature.HasField("lane") and len(feature.lane.polyline) >= 2:
            xy = np.array([(p.x, p.y) for p in feature.lane.polyline])
            if np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1) < 1e-6):
                continue
            distance = np.linalg.norm(project(xy, ego.position)[1] - ego.position)
            lane_candidates.append(
                (0 if lane_id in route.lane_ids else 1, distance, lane_id)
            )
    for slot, (_, _, lane_id) in enumerate(sorted(lane_candidates)[:16]):
        lane = features_by_id[lane_id].lane
        xy = np.array([(p.x, p.y) for p in lane.polyline])
        xy = sample(xy, np.linspace(0, arc(xy)[-1], 20))
        tangents = np.array([project(xy, p)[2] for p in xy])
        lane_values[slot] = np.c_[
            (xy - ego.position) @ r,
            tangents @ r,
            np.full(20, lane.speed_limit_mph * 0.44704),
            np.full(20, float(lane_id in route.lane_ids)),
            np.full(20, float(lane_id == command.draft.target_lane_id)),
        ]
        lane_valid[slot], items["lanes"][slot] = True, True
        if not np.isfinite(lane.speed_limit_mph) or lane.speed_limit_mph <= 0:
            lane_valid[slot, :, 4] = False
            lane_values[slot, :, 4] = 0
    write("lanes", lane_values, lane_valid, [100, 100, 1, 1, 30, 1, 1])
    boundary_values = np.zeros((16, 20, 5))
    boundary_valid = np.zeros_like(boundary_values, bool)
    # Expose the original referenced map features, including road-edge type.
    for slot, feature_id in enumerate(route.boundary_ids[:16]):
        feature = features_by_id[feature_id]
        kind = feature.WhichOneof("feature_data")
        xy = np.array([(p.x, p.y) for p in getattr(feature, kind).polyline])
        xy = sample(xy, np.linspace(0, arc(xy)[-1], 20))
        tangents = np.array([project(xy, p)[2] for p in xy])
        boundary_values[slot] = np.c_[
            (xy - ego.position) @ r,
            tangents @ r,
            np.full(20, float(kind == "road_edge")),
        ]
        boundary_valid[slot], items["boundaries"][slot] = True, True
    write("boundaries", boundary_values, boundary_valid, [100, 100, 1, 1, 1])
    # Signalized routes are excluded at reset; signal fields remain invalid.
    draft = command.draft
    command_values = np.zeros(19)
    command_valid = np.ones(19, bool)
    command_values[list(Maneuver).index(draft.maneuver)] = 1
    command_values[5 + list(Direction).index(draft.direction)] = 1
    command_values[8] = draft.desired_speed_mps
    for i, value in ((9, draft.time_gap_s), (10, draft.minimum_gap_m)):
        command_values[i] = value or 0.0
        command_valid[i] = value is not None
    command_valid[11:15] = False
    if draft.target_pose_world is not None:
        pose = draft.target_pose_world
        command_values[11:15] = [
            *((pose[:2] - ego.position) @ r),
            np.sin(pose[2] - ego.yaw),
            np.cos(pose[2] - ego.yaw),
        ]
        command_valid[11:15] = True
    command_values[15:19] = [
        draft.target_pose_world is not None,
        draft.target_agent_id is not None,
        draft.target_lane_id is not None,
        max(0, command.valid_until_tick - e.tick) * DT,
    ]
    write(
        "command",
        command_values,
        command_valid,
        [1] * 8 + [30, 20, 100, 100, 100, 1, 1, 1, 1, 1, 20],
    )
    execution = np.zeros(25)
    execution_valid = np.ones(25, bool)
    execution[list(Stage).index(e.feedback.stage)] = 1
    execution[10 + list(Source).index(e.feedback.source)] = 1
    execution[17:] = [
        e.failure_streak,
        (e.tick - e.old_plan.proposal.generated_tick) * DT if e.old_plan else 0,
        1,
        e.feedback.execute_authorized,
        e.failure_streak >= 3,
        0,
        0,
        0,
    ]
    execution_valid[18] = e.old_plan is not None
    execution_valid[22:25] = False  # no lateral completion lifecycle in this milestone
    write(
        "execution",
        execution,
        execution_valid,
        [1] * 17 + [3, 20, 1, 1, 1, 20, 100, 20],
    )
    if e.old_plan:
        times = (e.tick + np.arange(1, 31)) * DT
        valid = times <= e.old_plan.end + 1e-9
        plan = np.zeros((30, 6))
        plan_mask = np.repeat(valid[:, None], 6, axis=1)
        for index in np.flatnonzero(valid):
            t = times[index]
            p, v, a = [e.old_plan.evaluate(t, d) for d in (0, 1, 2)]
            speed = np.linalg.norm(v)
            curvature = (v[0] * a[1] - v[1] * a[0]) / speed**3 if speed >= 0.01 else 0.0
            plan[index] = [
                *((p - ego.position) @ r),
                *(v @ r),
                curvature,
                t - e.tick * DT,
            ]
            plan_mask[index, 4] = speed >= 0.01
        items["previous_plan"] = valid
        write("previous_plan", plan, plan_mask, [100, 100, 30, 30, 0.3, 20])
    m = e.memory
    write(
        "controller_memory",
        [
            m.previous_e_y,
            m.previous_e_v,
            m.filtered_de_y,
            m.filtered_de_v,
            m.last_a,
            m.last_kappa,
        ],
        [m.errors_valid] * 4 + [m.applied_control_valid] * 2,
        [100, 30, 30, 6, 6, 0.3],
    )
    e.clipped_features = clipped
    return CoreObservation(features, masks, items)
