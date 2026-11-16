"""Conservative lane corridors built from WOMD boundary feature references.

There is no nominal-width fallback. Missing/ambiguous boundary coverage rejects
the scene. Geometry stays in float64; no future ego trajectory defines a route.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


class ResetRejected(ValueError):
    """Explicit data/ODD exclusion, distinct from a failed policy episode."""


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def arc(xy):
    return np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]


def project(xy, point):
    d = np.diff(xy, axis=0)
    norm2 = np.sum(d * d, axis=1)
    if len(d) == 0 or np.any(norm2 < 1e-12):
        raise ResetRejected("DEGENERATE_POLYLINE")
    f = np.clip(np.sum((point - xy[:-1]) * d, axis=1) / norm2, 0, 1)
    positions = xy[:-1] + f[:, None] * d
    i = int(np.argmin(np.sum((positions - point) ** 2, axis=1)))
    tangent = d[i] / np.sqrt(norm2[i])
    return float(arc(xy)[i] + f[i] * np.sqrt(norm2[i])), positions[i], tangent


def sample(xy, distances):
    s = arc(xy)
    return np.stack([np.interp(distances, s, xy[:, k]) for k in (0, 1)], axis=-1)


def box(position, yaw, length, width, margin=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float)
    corners *= np.array([length / 2 + margin, width / 2 + margin])
    return position + corners @ np.array([[c, s], [-s, c]])


def overlaps(a, b):
    """Separating-axis OBB collision; touching is collision."""
    edges = np.concatenate([np.diff(a, axis=0)[:2], np.diff(b, axis=0)[:2]])
    axes = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    ap, bp = a @ axes.T, b @ axes.T
    return bool(
        np.all(
            (ap.max(axis=0) >= bp.min(axis=0) - 1e-9)
            & (bp.max(axis=0) >= ap.min(axis=0) - 1e-9)
        )
    )


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _crossings(a, b, c, d):
    ab, cd = b - a, d - c
    return (_cross(ab, c - a) * _cross(ab, d - a) < -1e-12) & (
        _cross(cd, a - c) * _cross(cd, b - c) < -1e-12
    )


def inside(polygon, points):
    return _inside_prepared(points, _prepare_polygon(polygon))


def _prepare_polygon(polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edge = b - a
    denom = edge[:, 1]
    slope = np.divide(edge[:, 0], denom, out=np.zeros_like(denom), where=denom != 0)
    norm2 = np.maximum(np.sum(edge * edge, axis=1), 1e-12)
    pad = 1e-8 + 32 * np.finfo(np.float64).eps * np.maximum(
        np.maximum(np.abs(a), np.abs(b)), 1
    )
    return a, b, edge, slope, norm2, np.minimum(a, b) - pad, np.maximum(a, b) + pad


def _inside_prepared(points, prepared):
    # Same ray / boundary tolerances and arithmetic as the scalar reference.
    a, b, edge, slope, norm2, low, high = prepared
    p = np.asarray(points)[:, None, :]
    crossing_y = (a[:, 1] > p[..., 1]) != (b[:, 1] > p[..., 1])
    x = a[:, 0] + (p[..., 1] - a[:, 1]) * slope
    ray = np.count_nonzero(crossing_y & (p[..., 0] < x), axis=1) % 2 != 0
    # Only points near an edge's bounding box can satisfy the original 1e-8
    # distance test. Padding makes this broad phase conservative; the final
    # distance/tolerance calculation itself is unchanged.
    rows, columns = np.nonzero(((p >= low) & (p <= high)).all(axis=-1))
    on_edge = np.zeros(len(p), bool)
    if len(rows):
        selected = p[rows, 0]
        f = np.clip(
            np.sum((selected - a[columns]) * edge[columns], axis=-1) / norm2[columns],
            0,
            1,
        )
        distance = np.linalg.norm(
            selected - (a[columns] + f[:, None] * edge[columns]), axis=-1
        )
        on_edge[rows[distance <= 1e-8]] = True
    return ray | on_edge


def contains_box(polygon, rectangle):
    return _contains_box_prepared(rectangle, _prepare_polygon(polygon))


def _contains_box_prepared(rectangle, prepared):
    if not _inside_prepared(rectangle, prepared).all():
        return False
    # Corners alone are insufficient for a concave corridor.
    a, b = prepared[:2]
    return not _crossings(
        rectangle[:, None, :], np.roll(rectangle, -1, axis=0)[:, None, :], a, b
    ).any()


def _rectangle_crossings(rectangles, prepared):
    a, b = prepared[:2]
    c, d = rectangles, np.roll(rectangles, -1, axis=1)
    low, high = prepared[5:]
    possible = (
        (np.maximum(c, d)[:, :, None, :] >= low)
        & (np.minimum(c, d)[:, :, None, :] <= high)
    ).all(axis=-1)
    rows, edges, columns = np.nonzero(possible)
    result = np.zeros(len(rectangles), bool)
    if len(rows):
        hit = _crossings(c[rows, edges], d[rows, edges], a[columns], b[columns])
        result[rows[hit]] = True
    return result


def _clean(xy):
    xy = np.asarray(xy, np.float64)
    if len(xy) < 2 or not np.isfinite(xy).all():
        raise ResetRejected("INVALID_MAP_GEOMETRY")
    return xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]]


def _boundary(lane, centers, features, side):
    """Respect lane index ranges; project only onto the referenced feature."""
    points = np.full_like(centers, np.nan)
    used = []
    for boundary in getattr(lane, side + "_boundaries"):
        start, end = int(boundary.lane_start_index), int(boundary.lane_end_index)
        if not 0 <= start <= end < len(centers):
            raise ResetRejected("INVALID_BOUNDARY_INDEX")
        feature = features.get(boundary.boundary_feature_id)
        if feature is None or feature.WhichOneof("feature_data") not in (
            "road_line",
            "road_edge",
        ):
            raise ResetRejected("MISSING_BOUNDARY_FEATURE")
        line = _clean(
            [
                (p.x, p.y)
                for p in getattr(feature, feature.WhichOneof("feature_data")).polyline
            ]
        )
        for i in range(start, end + 1):
            _, closest, _ = project(line, centers[i])
            if (
                np.isfinite(points[i]).all()
                and np.linalg.norm(points[i] - closest) > 0.25
            ):
                raise ResetRejected("AMBIGUOUS_BOUNDARY_JOIN")
            points[i] = closest
        used.append(int(feature.id))
    return points, used


def _lane_sections(lane, features):
    centers = np.array([(p.x, p.y) for p in lane.polyline])
    if len(centers) < 2 or len(_clean(centers)) != len(centers):
        raise ResetRejected("DEGENERATE_LANE_CENTER")
    left, lids = _boundary(lane, centers, features, "left")
    right, rids = _boundary(lane, centers, features, "right")
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
    ends = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
    return [
        (centers[a:b], left[a:b], right[a:b], lids + rids, a, b)
        for a, b in zip(starts, ends)
        if b - a >= 2
    ]


@dataclass(frozen=True)
class Route:
    route_id: str
    lane_ids: tuple[int, ...]
    center: np.ndarray
    left: np.ndarray
    right: np.ndarray
    polygon: np.ndarray
    speeds: np.ndarray
    start_s: float
    goal_s: float
    boundary_ids: tuple[int, ...]

    def __post_init__(self):
        # Caches must never outlive a mutation of their source geometry.
        for name in ("center", "left", "right", "polygon", "speeds"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @cached_property
    def _arc(self):
        return arc(self.center)

    @cached_property
    def _segments(self):
        delta = np.diff(self.center, axis=0)
        norm2 = np.sum(delta * delta, axis=1)
        return delta, norm2, np.sqrt(norm2)

    @cached_property
    def _polygon(self):
        return _prepare_polygon(self.polygon)

    def project(self, position):
        d, norm2, lengths = self._segments
        f = np.clip(np.sum((position - self.center[:-1]) * d, axis=1) / norm2, 0, 1)
        positions = self.center[:-1] + f[:, None] * d
        i = int(np.argmin(np.sum((positions - position) ** 2, axis=1)))
        return float(self._arc[i] + f[i] * lengths[i]), positions[i], d[i] / lengths[i]

    def project_many(self, positions):
        d, norm2, lengths = self._segments
        output = []
        for first in range(0, len(positions), 64):
            points = np.asarray(positions[first : first + 64])[:, None, :]
            f = np.clip(np.sum((points - self.center[:-1]) * d, axis=-1) / norm2, 0, 1)
            closest = self.center[:-1] + f[..., None] * d
            i = np.argmin(np.sum((closest - points) ** 2, axis=-1), axis=1)
            row = np.arange(len(points))
            output.append(
                (
                    self._arc[i] + f[row, i] * lengths[i],
                    closest[row, i],
                    d[i] / lengths[i, None],
                )
            )
        return tuple(np.concatenate([row[j] for row in output]) for j in range(3))

    def speed_limits_at(self, distances):
        i = np.minimum(np.searchsorted(self._arc, distances), len(self.speeds) - 1)
        return np.minimum(self.speeds[np.maximum(i - 1, 0)], self.speeds[i])

    def contains_many(self, rectangles):
        results = []
        a, b = self._polygon[:2]
        for first in range(0, len(rectangles), 64):
            rect = np.asarray(rectangles[first : first + 64])
            corners = (
                _inside_prepared(rect.reshape(-1, 2), self._polygon)
                .reshape(-1, 4)
                .all(axis=1)
            )
            crossing = _rectangle_crossings(rect, self._polygon)
            results.append(corners & ~crossing)
        return np.concatenate(results)

    @property
    def length(self):
        return float(self._arc[-1])

    def speed_limit(self, position):
        s, _, _ = self.project(position)
        index = min(int(np.searchsorted(self._arc, s)), len(self.speeds) - 1)
        return float(min(self.speeds[max(index - 1, 0) : index + 1]))

    def contains(self, position, yaw, length, width, margin=0.0):
        return _contains_box_prepared(
            box(position, yaw, length, width, margin), self._polygon
        )

    def coverage_exited(self, rectangle):
        """Longitudinal exit of the constructed contiguous route-map coverage.

        Called only after corridor containment fails. Lateral exits remain
        corridor violations; nearby curved segments must not trigger an end cap.
        """
        for point in rectangle:
            s, nearest, tangent = self.project(point)
            beyond = float(np.dot(point - nearest, tangent))
            if (s <= 1e-9 and beyond < -1e-6) or (
                s >= self.length - 1e-9 and beyond > 1e-6
            ):
                return True
        return False


def build_route(scenario, ego, frame):
    features = {int(f.id): f for f in scenario.map_features}
    lanes = {i: f.lane for i, f in features.items() if f.HasField("lane")}
    candidates = []
    for lane_id, lane in lanes.items():
        if len(lane.polyline) < 2:
            continue
        xy = np.array([(p.x, p.y) for p in lane.polyline])
        try:
            s, nearest, tangent = project(xy, ego.position)
        except ResetRejected:
            continue
        distance = np.linalg.norm(nearest - ego.position)
        heading = abs(wrap(np.arctan2(tangent[1], tangent[0]) - ego.yaw))
        if distance <= 8 and heading <= np.pi / 6:
            candidates.append((distance, heading, lane_id, s))
    valid = []
    reasons = []
    for _, _, lane_id, _ in sorted(candidates):
        try:
            lane = lanes[lane_id]
            for section in _lane_sections(lane, features):
                center, left, right, _, _, _ = section
                if contains_box(
                    np.concatenate([left, right[::-1]]),
                    box(ego.position, ego.yaw, ego.length, ego.width, 0.1),
                ):
                    valid.append((lane_id, section))
        except ResetRejected as error:
            reasons.append(str(error))
    if len(valid) != 1:
        reason = "AMBIGUOUS_START_LANE" if len(valid) > 1 else "NO_VALID_START_CORRIDOR"
        raise ResetRejected(
            reason + (":" + ",".join(sorted(set(reasons))) if reasons else "")
        )
    lane_id, first_section = valid[0]
    center, left, right, speeds, lane_ids, boundaries = [], [], [], [], [], []
    start_s = goal_s = None
    while True:
        if lane_id in lane_ids or lane_id not in lanes:
            raise ResetRejected("INVALID_ROUTE_TOPOLOGY")
        lane = lanes[lane_id]
        if len(lane.entry_lanes) > 1 or lane.interpolating:
            raise ResetRejected("MERGE_OR_INTERSECTION_OUTSIDE_INITIAL_ODD")
        speed = float(lane.speed_limit_mph) * 0.44704
        if not np.isfinite(speed) or speed <= 0:
            raise ResetRejected("UNKNOWN_SPEED_LIMIT")
        sections = _lane_sections(lane, features)
        if not lane_ids:
            section = first_section
        else:
            sections = [
                x for x in sections if np.linalg.norm(x[0][0] - center[-1]) <= 1.0
            ]
            if len(sections) != 1:
                raise ResetRejected("INCOMPLETE_BOUNDARY_COVERAGE")
            section = sections[0]
        xy, l, r, boundary_ids, start_index, end_index = section
        tangents = np.gradient(xy, axis=0)
        if np.any(_cross(tangents, l - xy) <= 0) or np.any(
            _cross(tangents, r - xy) >= 0
        ):
            raise ResetRejected("BOUNDARY_SIDE_MISMATCH")
        if np.any(np.linalg.norm(l - r, axis=1) < ego.width + 0.2):
            raise ResetRejected("INSUFFICIENT_CORRIDOR_WIDTH")
        if center:
            if (
                max(
                    np.linalg.norm(center[-1] - xy[0]),
                    np.linalg.norm(left[-1] - l[0]),
                    np.linalg.norm(right[-1] - r[0]),
                )
                > 1.0
            ):
                raise ResetRejected("DISCONNECTED_CORRIDOR")
            if np.linalg.norm(center[-1] - xy[0]) < 1e-6:
                xy, l, r = xy[1:], l[1:], r[1:]
        center.extend(xy)
        left.extend(l)
        right.extend(r)
        speeds.extend([speed] * len(xy))
        lane_ids.append(lane_id)
        boundaries.extend(boundary_ids)
        if start_s is None:
            start_s = project(np.array(center), ego.position)[0]
            goal_s = start_s + float(np.clip(min(speed, 10.0) * 8, 30, 80))
        if arc(np.array(center))[-1] >= goal_s + 45 + ego.length:
            break
        if end_index < len(lane.polyline) - 1:
            raise ResetRejected("INCOMPLETE_BOUNDARY_COVERAGE")
        if len(lane.exit_lanes) != 1:
            raise ResetRejected("AMBIGUOUS_OR_SHORT_ROUTE")
        lane_id = int(lane.exit_lanes[0])
    if scenario.dynamic_map_states:
        if any(
            int(light.lane) in lane_ids
            for light in scenario.dynamic_map_states[frame].lane_states
        ):
            raise ResetRejected("SIGNALIZED_ROUTE_OUTSIDE_INITIAL_ODD")
    for feature in features.values():
        if feature.HasField("stop_sign") and set(feature.stop_sign.lane).intersection(
            lane_ids
        ):
            raise ResetRejected("STOP_SIGN_OUTSIDE_INITIAL_ODD")
    left, right = _clean(left), _clean(right)
    polygon = np.concatenate([left, right[::-1]])
    for i in range(len(polygon)):
        others = np.arange(i + 2, len(polygon))
        if i == 0:
            others = others[others != len(polygon) - 1]
        if _crossings(
            polygon[i],
            polygon[(i + 1) % len(polygon)],
            polygon[others],
            polygon[(others + 1) % len(polygon)],
        ).any():
            raise ResetRejected("SELF_INTERSECTING_CORRIDOR")
    return Route(
        f"route:{scenario.scenario_id}:{lane_ids[0]}",
        tuple(lane_ids),
        np.array(center),
        left,
        right,
        polygon,
        np.array(speeds),
        start_s,
        goal_s,
        tuple(sorted(set(boundaries))),
    )
