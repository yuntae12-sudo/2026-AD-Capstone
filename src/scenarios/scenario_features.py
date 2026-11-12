"""Legacy single-frame interaction features for accepted merge candidates.

Given an accepted ``MergeDiagnostic`` (see ``merge_detector.py``), this
module identifies the target-lane Front/Rear vehicles at a chosen
frame and computes the gap / relative speed / TTC / remaining-merge
-distance / traffic-density features the paper's planned 8D common
state is built from (see project plan section 19):

    v_e     <- ego_longitudinal_speed_mps (TARGET-LANE longitudinal
               component, not raw speed magnitude -- see fix below)
    d_m     <- merge_distance_m
    d_f     <- front_gap_m
    Delta v_f <- front_relative_speed_mps
    d_r     <- rear_gap_m
    Delta v_r <- rear_relative_speed_mps
    TTC_f   <- front_ttc_s
    TTC_r   <- rear_ttc_s

This matcher is suitable for producing online observation candidates, but its
centre-point/one-frame result is not proof that a vehicle occupies the target
lane and must not be used as a ground-truth merge label. Offline labels must
pass :mod:`src.scenarios.physical_merge_validation`, which checks the bounding
box, trajectory, heading, identity persistence, and motion class.

This module does NOT build that final policy observation class (that
is Phase 3's job) -- it only extracts these raw, physically
interpretable values with no normalization or clipping (per project
plan section 5), reusing this same code so Phase 3 does not have to
reimplement Front/Rear selection or gap/TTC geometry.

Longitudinal velocity fix: every speed used here (ego's and each
Front/Rear candidate's) is the vehicle's velocity projected onto the
TARGET LANE's local unit tangent at that vehicle's own projection point
-- v_long = [vx, vy] . tangent -- not ``hypot(vx, vy)``. Raw speed
magnitude would include any lateral velocity component (e.g. a vehicle
not perfectly aligned with the lane, or mid-lane-change), which is not
physically meaningful for a longitudinal gap-closing interaction.

Event-time note: at a single fixed frame (typically the transition
frame), this module only reads state at that frame and does not depend
on future frames itself. However, WHICH frame is chosen to extract
features at (e.g. Commit D's real-WOMD inspection script uses the
transition frame of an already-accepted merge candidate) can itself be
an offline dataset-construction decision informed by future
information (the merge was already confirmed using post-transition
frames in merge_detector.py). Phase 3's online policy must not reuse
that selection logic -- it would need its own current-frame-only
trigger for when to compute these features.
"""

import dataclasses
from typing import List, Optional

import numpy as np
import yaml

from src.scenarios.lane_geometry import (
    LanePolyline,
    project_point_to_polyline,
    project_point_to_polyline_signed,
)

# Per Phase 0's OBJECT_TYPE_NAMES (scripts/run_scene.py): 1 = VEHICLE.
# Front/Rear candidates must be actual vehicles, not pedestrians/
# cyclists/other, since this is a vehicle-following interaction.
VEHICLE_OBJECT_TYPE = 1


@dataclasses.dataclass(frozen=True)
class AgentSelectionConfig:
    """Thresholds for target-lane Front/Rear candidate selection.

    See configs/merge.yaml `agent_selection` and `traffic` for
    field descriptions and rationale.
    """

    max_target_lane_lateral_distance_m: float
    max_target_lane_heading_difference_deg: float
    max_distance_m: float
    density_radius_m: float


@dataclasses.dataclass(frozen=True)
class InteractionFeatures:
    """Target-lane interaction features at a single frame.

    Convention (project plan sections 6-7, 14-17):
        - All speeds used for these interaction features -- ego's own
          and every Front/Rear candidate's -- are the TARGET-LANE
          LONGITUDINAL component (velocity projected onto the target
          lane's local unit tangent at each vehicle's own projection),
          NOT raw speed magnitude ``hypot(vx, vy)`` (project plan
          section 16 fix). ``ego_longitudinal_speed_mps`` is named
          explicitly to make this unambiguous for Phase 3 reuse -- see
          the 8D common state mapping at the top of this module: v_e is
          this longitudinal speed, not ego's raw ground speed.
        - front_gap_m / rear_gap_m are bumper-to-bumper (vehicle
          lengths subtracted), not center-to-center, and are allowed to
          be <= 0 (physical overlap / already-collided), never clamped
          positive.
        - front_relative_speed_mps = v_ego_long - v_front_long
          (positive = closing).
        - rear_relative_speed_mps = v_rear_long - v_ego_long (positive
          = closing) -- note this is NOT v_ego_long - v_rear_long; both
          conventions are defined so that positive always means
          closing, per project plan section 6/15.
        - front_ttc_s / rear_ttc_s = gap / closing_speed (both
          longitudinal), only when closing_speed > 0; otherwise inf
          (non-closing). If gap <= 0 (already overlapping), ttc is 0.0.
          No clipping to any maximum is applied here -- that belongs to
          later normalization (Phase 3), not this raw feature
          extraction.
        - With no Front/Rear candidate: vehicle_id=None, gap=None,
          relative_speed=None, ttc=inf.
    """

    frame_index: int

    ego_longitudinal_speed_mps: float
    merge_distance_m: float

    front_vehicle_id: Optional[int]
    front_gap_m: Optional[float]
    front_relative_speed_mps: Optional[float]
    front_ttc_s: float

    rear_vehicle_id: Optional[int]
    rear_gap_m: Optional[float]
    rear_relative_speed_mps: Optional[float]
    rear_ttc_s: float

    traffic_density: int


def load_agent_selection_config(config_path: str) -> AgentSelectionConfig:
    """Loads the `agent_selection` + `traffic` sections of a Phase 1
    merge config YAML into one ``AgentSelectionConfig``.
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}

    agent_selection_raw = dict(raw.get("agent_selection", {}))
    traffic_raw = raw.get("traffic", {})

    agent_selection_raw["density_radius_m"] = traffic_raw.get(
        "density_radius_m"
    )

    return AgentSelectionConfig(**agent_selection_raw)


def _wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _longitudinal_speed(
    vel_x: float, vel_y: float, target_heading_rad: float
) -> float:
    """Projects a velocity vector onto the target lane's local unit
    tangent (project plan section 16): v_long = [vx, vy] . tangent,
    NOT hypot(vx, vy). Using raw speed magnitude would include any
    lateral velocity component (e.g. a vehicle drifting sideways within
    its lane, or one whose heading isn't perfectly lane-aligned), which
    is not the component relevant to a longitudinal
    gap-closing/following interaction.
    """

    tangent = np.array(
        [np.cos(target_heading_rad), np.sin(target_heading_rad)]
    )

    return float(np.dot([vel_x, vel_y], tangent))


def _compute_ttc(gap_m: float, closing_speed_mps: float) -> float:
    """TTC = gap / closing_speed, only when closing_speed > 0.

    In this legacy feature path, gap <= 0 -> 0.0 (project plan section 17:
    "preferably 0 for already-overlapping/collision-state
    interpretation"). This is only a 1-D centreline-projection result; it is
    not evidence of physical overlap until both bounding boxes have passed the
    target-lane validator. Otherwise, non-closing (closing_speed <= 0) ->
    inf. No maximum clipping is applied.
    """

    if gap_m <= 0.0:
        return 0.0

    if closing_speed_mps <= 0.0:
        return float("inf")

    return float(gap_m / closing_speed_mps)


def find_target_lane_front_rear(
    target_polyline: LanePolyline,
    ego_s_m: float,
    ego_id: int,
    frame_index: int,
    object_ids: np.ndarray,
    object_types: np.ndarray,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    config: AgentSelectionConfig,
):
    """Finds legacy Front/Rear *candidates* at one frame.

    This intentionally preserves the frozen v1/v2 online feature contract.
    Do not promote its output directly to a physical ground-truth label.

    Candidate surrounding vehicles are projected onto the TARGET lane
    (project plan section 13: never ego-local or global x/y) and must
    be: valid at this frame, a vehicle (``VEHICLE_OBJECT_TYPE``), not
    ego itself, within ``config.max_target_lane_lateral_distance_m`` of
    the target lane centerline, within
    ``config.max_target_lane_heading_difference_deg`` of the target
    lane's local heading at the candidate's own projection (fix: lane
    -lateral-distance filtering alone can select a crossing or opposite
    -direction vehicle at an intersection that happens to sit close to
    the target centerline -- see
    tests/scenarios/test_scenario_features.py's crossing/opposite
    -direction cases), and within ``config.max_distance_m`` of ego in
    target-lane arc length. Among survivors, Front is the smallest
    positive (s_vehicle - s_ego); Rear is the largest negative.

    Longitudinal ordering fix (Stage B-0): both `ego_s_m` and each
    candidate's own longitudinal position are computed with
    ``project_point_to_polyline_signed`` (SIGNED/unclamped arc length),
    not ``project_point_to_polyline`` (which clamps to the polyline's
    own domain). Without this, ego and any candidate vehicle upstream
    of the target lane's own start point (common at an early
    pre-merge reference frame, before ego has entered the target lane)
    would both clamp to the identical `arc_length_m=0.0`, making
    `relative_s == 0.0` for two physically distinct vehicles and
    silently excluding a real rear (or front) vehicle from selection.
    The lateral-distance and heading-difference gates below still use
    the ordinary (clamped) projection for their own filtering, since
    those checks are about proximity to the polyline's physical
    extent, not longitudinal order.

    Args:
        target_polyline: the target lane to project onto.
        ego_s_m: ego's own SIGNED target-lane longitudinal position
            (see ``project_point_to_polyline_signed``), not the
            polyline-domain-clamped ``arc_length_m``.
        ego_id: ego's object id (excluded from candidates).
        frame_index: which frame of `object_ids`/`x`/`y`/`valid`/`yaw`
            to use (all object-indexed arrays here are assumed already
            sliced to this frame, i.e. shape (num_objects,);
            frame_index is only carried through for any caller
            bookkeeping needs).
        object_ids, object_types, valid, x, y, yaw: (num_objects,)
            arrays at this frame.
        config: thresholds.

    Returns:
        (front_id, front_s, rear_id, rear_s): ids are None and s values
        are None when no candidate survives on that side.
    """

    del frame_index  # arrays are already frame-sliced by the caller

    front_id, front_s = None, None
    rear_id, rear_s = None, None
    best_front_gap_s = None
    best_rear_gap_s = None

    for i in range(object_ids.shape[0]):

        if not valid[i]:
            continue

        candidate_id = int(object_ids[i])

        if candidate_id == ego_id:
            continue

        if int(object_types[i]) != VEHICLE_OBJECT_TYPE:
            continue

        projection = project_point_to_polyline(
            target_polyline, float(x[i]), float(y[i])
        )

        lateral_distance = abs(projection["lateral_distance_m"])

        if lateral_distance > config.max_target_lane_lateral_distance_m:
            continue

        heading_difference_deg = abs(
            _wrap_angle_deg(
                np.degrees(projection["heading_rad"]) - np.degrees(yaw[i])
            )
        )

        if (
            heading_difference_deg
            > config.max_target_lane_heading_difference_deg
        ):
            continue

        # SIGNED projection for longitudinal ordering only (Stage B-0
        # fix) -- lateral/heading gating above already used the
        # ordinary clamped projection, which is correct for those
        # proximity checks.
        signed_projection = project_point_to_polyline_signed(
            target_polyline, float(x[i]), float(y[i])
        )
        candidate_s = signed_projection["arc_length_m"]

        relative_s = candidate_s - ego_s_m

        if abs(relative_s) > config.max_distance_m:
            continue

        if relative_s > 0.0:
            if best_front_gap_s is None or relative_s < best_front_gap_s:
                best_front_gap_s = relative_s
                front_id = candidate_id
                front_s = candidate_s
        elif relative_s < 0.0:
            if best_rear_gap_s is None or relative_s > best_rear_gap_s:
                best_rear_gap_s = relative_s
                rear_id = candidate_id
                rear_s = candidate_s

    return front_id, front_s, rear_id, rear_s


def compute_traffic_density(
    ego_x: float,
    ego_y: float,
    object_ids: np.ndarray,
    object_types: np.ndarray,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ego_id: int,
    config: AgentSelectionConfig,
) -> int:
    """Counts valid vehicle agents within `config.density_radius_m` of
    ego (project plan section 18: a simple analysis/context variable,
    not part of the planned 8D policy state -- deliberately not
    over-engineered).
    """

    count = 0

    for i in range(object_ids.shape[0]):

        if not valid[i]:
            continue

        if int(object_ids[i]) == ego_id:
            continue

        if int(object_types[i]) != VEHICLE_OBJECT_TYPE:
            continue

        distance = float(np.hypot(x[i] - ego_x, y[i] - ego_y))

        if distance <= config.density_radius_m:
            count += 1

    return count


def extract_interaction_features(
    frame_index: int,
    target_polyline: LanePolyline,
    merge_distance_m: float,
    ego_id: int,
    ego_x: float,
    ego_y: float,
    ego_vel_x: float,
    ego_vel_y: float,
    ego_length_m: float,
    object_ids: np.ndarray,
    object_types: np.ndarray,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    vel_x: np.ndarray,
    vel_y: np.ndarray,
    length: np.ndarray,
    config: AgentSelectionConfig,
) -> InteractionFeatures:
    """Computes target-lane interaction features for ego at one frame.

    All object-indexed arrays (object_ids, object_types, valid, x, y,
    yaw, vel_x, vel_y, length) must already be sliced to `frame_index`
    (shape (num_objects,)).

    Args:
        frame_index: the frame these features are computed at.
        target_polyline: the merge target lane.
        merge_distance_m: d_m, already computed by
            ``merge_detector.compute_remaining_merge_distance`` (in
            SOURCE-lane arc length, per project plan section 12) --
            this function does not recompute it, since target-lane
            projection is not the right coordinate frame for d_m. It
            is passed through here only so a caller can fill in
            ``InteractionFeatures.merge_distance_m`` from a single
            call; see ``scripts/inspect_merge_candidate.py`` for the
            intended call pattern (compute d_m first, then this).
        ego_id, ego_x, ego_y, ego_vel_x, ego_vel_y, ego_length_m: ego's
            own state at this frame.
        object_ids..length: (num_objects,) arrays at this frame.
        config: thresholds.

    Returns:
        InteractionFeatures.
    """

    ego_projection = project_point_to_polyline(
        target_polyline, ego_x, ego_y
    )
    ego_longitudinal_speed_mps = _longitudinal_speed(
        ego_vel_x, ego_vel_y, ego_projection["heading_rad"]
    )

    # Signed (unclamped) target-lane longitudinal coordinate for
    # relative front/rear ORDERING only (Stage B-0 fix). The clamped
    # `ego_projection["arc_length_m"]` above is deliberately kept for
    # `ego_longitudinal_speed_mps`'s heading and is otherwise unused
    # here -- see `project_point_to_polyline_signed`'s docstring for
    # why the clamped value corrupts front/rear selection whenever ego
    # (or a candidate) projects before the target lane's own start.
    ego_s_signed = project_point_to_polyline_signed(
        target_polyline, ego_x, ego_y
    )["arc_length_m"]

    front_id, front_s, rear_id, rear_s = find_target_lane_front_rear(
        target_polyline,
        ego_s_signed,
        ego_id,
        frame_index,
        object_ids,
        object_types,
        valid,
        x,
        y,
        yaw,
        config,
    )

    front_gap_m = None
    front_relative_speed_mps = None
    front_ttc_s = float("inf")

    if front_id is not None:
        front_index = int(np.flatnonzero(object_ids == front_id)[0])
        front_length = float(length[front_index])
        front_projection = project_point_to_polyline(
            target_polyline,
            float(x[front_index]),
            float(y[front_index]),
        )
        front_longitudinal_speed = _longitudinal_speed(
            float(vel_x[front_index]),
            float(vel_y[front_index]),
            front_projection["heading_rad"],
        )

        front_gap_m = float(
            (front_s - ego_s_signed) - 0.5 * (front_length + ego_length_m)
        )
        front_relative_speed_mps = float(
            ego_longitudinal_speed_mps - front_longitudinal_speed
        )
        front_ttc_s = _compute_ttc(front_gap_m, front_relative_speed_mps)

    rear_gap_m = None
    rear_relative_speed_mps = None
    rear_ttc_s = float("inf")

    if rear_id is not None:
        rear_index = int(np.flatnonzero(object_ids == rear_id)[0])
        rear_length = float(length[rear_index])
        rear_projection = project_point_to_polyline(
            target_polyline,
            float(x[rear_index]),
            float(y[rear_index]),
        )
        rear_longitudinal_speed = _longitudinal_speed(
            float(vel_x[rear_index]),
            float(vel_y[rear_index]),
            rear_projection["heading_rad"],
        )

        rear_gap_m = float(
            (ego_s_signed - rear_s) - 0.5 * (rear_length + ego_length_m)
        )
        rear_relative_speed_mps = float(
            rear_longitudinal_speed - ego_longitudinal_speed_mps
        )
        rear_ttc_s = _compute_ttc(rear_gap_m, rear_relative_speed_mps)

    traffic_density = compute_traffic_density(
        ego_x,
        ego_y,
        object_ids,
        object_types,
        valid,
        x,
        y,
        ego_id,
        config,
    )

    return InteractionFeatures(
        frame_index=frame_index,
        ego_longitudinal_speed_mps=ego_longitudinal_speed_mps,
        merge_distance_m=merge_distance_m,
        front_vehicle_id=front_id,
        front_gap_m=front_gap_m,
        front_relative_speed_mps=front_relative_speed_mps,
        front_ttc_s=front_ttc_s,
        rear_vehicle_id=rear_id,
        rear_gap_m=rear_gap_m,
        rear_relative_speed_mps=rear_relative_speed_mps,
        rear_ttc_s=rear_ttc_s,
        traffic_density=traffic_density,
    )
