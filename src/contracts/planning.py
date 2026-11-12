"""Schema 3 records. These do not change the legacy four-action environment.

World coordinates reference the vehicle box center. Only the environment owns
the clock and only its execution manager issues accepted command IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
from types import UnionType
from typing import get_args, get_origin, get_type_hints

import numpy as np

SCHEMA_VERSION = 3
DISTRIBUTION_VERSION = "correlated_xy_gaussian_accel_anchor_v2"
DT = 0.1
FUTURE_POINTS = 30
HORIZON = 3.0
BEHAVIOR_INTERVAL = 5
COMMAND_TTL = 10


class ContractError(ValueError):
    """Malformed boundary payload; never a fabricated learning transition."""


class Maneuver(str, Enum):
    KEEP = "KEEP"
    FOLLOW = "FOLLOW"
    MERGE = "MERGE"
    AVOID = "AVOID"
    STOP = "STOP"


class Direction(str, Enum):
    AUTO = "AUTO"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class ExecutionRequest(str, Enum):
    PREPARE = "PREPARE"
    EXECUTE = "EXECUTE"
    CANCEL = "CANCEL"


class Stage(str, Enum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    WAITING = "WAITING"
    READY = "READY"
    COMMITTED = "COMMITTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    HOLD = "HOLD"


class Source(str, Enum):
    NONE = "NONE"
    NEW = "NEW"
    REUSED = "REUSED"
    BRAKE = "BRAKE"
    PREPARE = "PREPARE"
    EMERGENCY = "EMERGENCY"
    HOLD = "HOLD"


class TaskKind(str, Enum):
    ROUTE = "ROUTE"
    STOP_AT_POSE = "STOP_AT_POSE"


class TerminalReason(str, Enum):
    COLLISION = "COLLISION"
    OFFROAD = "OFFROAD"
    CORRIDOR_VIOLATION = "CORRIDOR_VIOLATION"
    MAP_COVERAGE_EXIT = "MAP_COVERAGE_EXIT"
    ROUTE_GOAL = "ROUTE_GOAL"
    TASK_LOST = "TASK_LOST"
    RECORD_END = "RECORD_END"
    TIME_LIMIT = "TIME_LIMIT"
    DATA_CORRUPTION = "DATA_CORRUPTION"
    STOP_GOAL = "STOP_GOAL"


class RejectionReason(str, Enum):
    STALE_OBSERVATION = "STALE_OBSERVATION"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    UNSUPPORTED_MODE = "UNSUPPORTED_MODE"
    MISSING_TARGET = "MISSING_TARGET"
    INVALID_CORRIDOR = "INVALID_CORRIDOR"
    EXPIRED_COMMAND = "EXPIRED_COMMAND"
    SWITCH_LOCKED = "SWITCH_LOCKED"
    INVALID_ACTION = "INVALID_ACTION"
    NONFINITE_POLICY = "NONFINITE_POLICY"
    ANCHOR_MISMATCH = "ANCHOR_MISMATCH"
    KINEMATIC_LIMIT = "KINEMATIC_LIMIT"
    PREDICTED_COLLISION = "PREDICTED_COLLISION"
    PREDICTED_OFFROAD = "PREDICTED_OFFROAD"
    COMMAND_VIOLATION = "COMMAND_VIOLATION"
    TRACKING_FAILURE = "TRACKING_FAILURE"
    OLD_PLAN_EXPIRED = "OLD_PLAN_EXPIRED"
    UNRESOLVED_SWEEP = "UNRESOLVED_SWEEP"
    POLICY_ERROR = "POLICY_ERROR"
    DUPLICATE_REQUEST_MISMATCH = "DUPLICATE_REQUEST_MISMATCH"


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{name} must be an integer")
    if not minimum <= value <= np.iinfo(np.int64).max:
        raise ContractError(f"{name} outside int64 range")


def finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ContractError(f"{name} must be finite numeric")


def text(value, name):
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a nonempty string")


def array(value, shape, name, dtype=np.float64):
    """Copy into immutable storage; reject accidental float32 World payloads."""
    if isinstance(value, np.ndarray) and value.dtype != np.dtype(dtype):
        raise ContractError(f"{name} must have dtype {np.dtype(dtype)}")
    original = np.asarray(value)
    if np.dtype(dtype).kind in "biu" and original.dtype.kind != np.dtype(dtype).kind:
        raise ContractError(f"{name} must contain {np.dtype(dtype)} values")
    a = np.asarray(value, dtype=dtype)
    if a.shape != shape or not np.isfinite(a).all():
        raise ContractError(f"{name} must have finite shape {shape}")
    return np.frombuffer(a.tobytes(order="C"), dtype=dtype).reshape(shape)


def _enum(value, kind):
    if not isinstance(value, kind):
        raise ContractError(f"Expected {kind.__name__}, got {value!r}")


@dataclass(frozen=True)
class ObservationToken:
    episode_uid: str
    scene_id: str
    sim_tick: int
    revision: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        text(self.episode_uid, "episode_uid")
        text(self.scene_id, "scene_id")
        for name in ("sim_tick", "revision", "schema_version"):
            integer(getattr(self, name), name)
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError("SCHEMA_MISMATCH")

    def require_current(self, current):
        if self != current:
            raise ContractError("STALE_OBSERVATION")


@dataclass(frozen=True)
class TaskSpec:
    kind: TaskKind
    goal_id: str
    route_goal_s: float
    stop_pose_world: np.ndarray | None = None

    def __post_init__(self):
        _enum(self.kind, TaskKind)
        text(self.goal_id, "goal_id")
        finite(self.route_goal_s, "route_goal_s")
        if self.route_goal_s < 0:
            raise ContractError("negative route goal")
        if self.stop_pose_world is not None:
            object.__setattr__(self, "stop_pose_world", array(self.stop_pose_world, (3,), "stop_pose_world"))
        if self.kind == TaskKind.STOP_AT_POSE and self.stop_pose_world is None:
            raise ContractError("STOP task requires a fixed goal pose")


@dataclass(frozen=True)
class CommandDraft:
    maneuver: Maneuver
    route_id: str
    source_lane_id: int
    target_lane_id: int | None
    target_agent_id: int | None
    target_pose_world: np.ndarray | None
    direction: Direction
    desired_speed_mps: float
    time_gap_s: float | None
    minimum_gap_m: float | None
    preferred_front_id: int | None
    preferred_rear_id: int | None
    allowed_corridor_id: str
    binding_policy_version: str

    def __post_init__(self):
        _enum(self.maneuver, Maneuver)
        _enum(self.direction, Direction)
        for name in ("route_id", "allowed_corridor_id", "binding_policy_version"):
            text(getattr(self, name), name)
        integer(self.source_lane_id, "source_lane_id")
        for name in ("target_lane_id", "target_agent_id", "preferred_front_id", "preferred_rear_id"):
            if getattr(self, name) is not None:
                integer(getattr(self, name), name)
        for name in ("desired_speed_mps", "time_gap_s", "minimum_gap_m"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
                if value < 0:
                    raise ContractError(f"negative {name}")
        if self.desired_speed_mps is None:
            raise ContractError("desired speed is required")
        if self.target_pose_world is not None:
            object.__setattr__(self, "target_pose_world", array(self.target_pose_world, (3,), "target_pose_world"))
        if self.maneuver == Maneuver.FOLLOW and any(x is None for x in (self.target_agent_id, self.time_gap_s, self.minimum_gap_m)):
            raise ContractError("FOLLOW requires a target and gap")
        if self.maneuver == Maneuver.MERGE and self.target_lane_id is None:
            raise ContractError("MERGE requires a target lane")
        if self.maneuver == Maneuver.AVOID and self.target_agent_id is None:
            raise ContractError("AVOID requires a target agent")
        if self.maneuver == Maneuver.STOP and self.target_pose_world is None:
            raise ContractError("STOP requires a target pose")


@dataclass(frozen=True)
class CommandRequest:
    request_id: str
    candidate_token: str | None
    command_draft: CommandDraft
    execution_request: ExecutionRequest
    provider_version: str

    def __post_init__(self):
        text(self.request_id, "request_id")
        text(self.provider_version, "provider_version")
        if self.candidate_token is not None:
            text(self.candidate_token, "candidate_token")
        if not isinstance(self.command_draft, CommandDraft):
            raise ContractError("CommandDraft required")
        _enum(self.execution_request, ExecutionRequest)


@dataclass(frozen=True)
class BehaviorCommand:
    command_id: int
    draft: CommandDraft
    created_tick: int
    valid_until_tick: int
    schema_version: int
    provider_version: str

    def __post_init__(self):
        integer(self.command_id, "command_id", 1)
        integer(self.created_tick, "created_tick")
        integer(self.valid_until_tick, "valid_until_tick")
        integer(self.schema_version, "schema_version")
        text(self.provider_version, "provider_version")
        if not isinstance(self.draft, CommandDraft) or self.schema_version != SCHEMA_VERSION:
            raise ContractError("SCHEMA_MISMATCH")
        if self.valid_until_tick <= self.created_tick:
            raise ContractError("invalid command lifetime")

    def is_valid_at(self, tick):
        integer(tick, "tick")
        return self.created_tick <= tick < self.valid_until_tick


@dataclass(frozen=True)
class Receipt:
    request_id: str
    accepted: bool
    reason: RejectionReason | None
    active_command_id: int | None
    revision: int

    def __post_init__(self):
        text(self.request_id, "request_id")
        integer(self.revision, "revision")
        if type(self.accepted) is not bool:
            raise ContractError("accepted must be bool")
        if self.reason is not None:
            _enum(self.reason, RejectionReason)
        if self.active_command_id is not None:
            integer(self.active_command_id, "active_command_id", 1)
        if self.accepted and self.reason is not None:
            raise ContractError("accepted receipt cannot have rejection reason")


@dataclass(frozen=True)
class WaypointProposal:
    observation_token: ObservationToken
    command_id: int
    proposal_id: str
    generated_tick: int
    anchor_world_xy: np.ndarray
    anchor_yaw: float
    anchor_velocity_world: np.ndarray
    anchor_acceleration_world: np.ndarray
    anchor_acceleration_valid: bool
    times: np.ndarray
    xy_world: np.ndarray
    raw_u: np.ndarray
    reference_point: str = "BOX_CENTER"
    model_version: str = "development"
    action_distribution_version: str = DISTRIBUTION_VERSION

    def __post_init__(self):
        if not isinstance(self.observation_token, ObservationToken):
            raise ContractError("ObservationToken required")
        integer(self.command_id, "command_id", 1)
        integer(self.generated_tick, "generated_tick")
        text(self.proposal_id, "proposal_id")
        text(self.model_version, "model_version")
        finite(self.anchor_yaw, "anchor_yaw")
        if not -np.pi <= self.anchor_yaw < np.pi:
            raise ContractError("yaw must lie in [-pi, pi)")
        if type(self.anchor_acceleration_valid) is not bool:
            raise ContractError("acceleration validity must be bool")
        for name, shape in (("anchor_world_xy", (2,)), ("anchor_velocity_world", (2,)),
                            ("anchor_acceleration_world", (2,)), ("times", (30,)),
                            ("xy_world", (30, 2)), ("raw_u", (30, 2))):
            object.__setattr__(self, name, array(getattr(self, name), shape, name))
        if not self.anchor_acceleration_valid and np.any(self.anchor_acceleration_world != 0):
            raise ContractError("invalid acceleration must be zero")
        if self.generated_tick != self.observation_token.sim_tick:
            raise ContractError("generated tick differs from observation")
        if not np.allclose(self.times, future_times(self.generated_tick), atol=1e-12, rtol=0):
            raise ContractError("times must be episode model time, future ticks 1..30")
        if self.reference_point != "BOX_CENTER" or self.action_distribution_version != DISTRIBUTION_VERSION:
            raise ContractError("SCHEMA_MISMATCH")
        decoded = decode_waypoints(self.raw_u, self.anchor_world_xy, self.anchor_yaw,
                                   self.anchor_velocity_world, self.anchor_acceleration_world)
        if np.max(np.abs(decoded - self.xy_world)) > 1e-7:
            raise ContractError("raw_u / xy_world mismatch")


@dataclass(frozen=True)
class ControlCommand:
    sim_tick: int
    acceleration_mps2: float
    curvature_per_m: float
    source: Source
    saturated: bool
    limit_override_reason: str | None = None

    def __post_init__(self):
        integer(self.sim_tick, "sim_tick")
        finite(self.acceleration_mps2, "acceleration_mps2")
        finite(self.curvature_per_m, "curvature_per_m")
        _enum(self.source, Source)
        if type(self.saturated) is not bool:
            raise ContractError("saturated must be bool")
        if self.limit_override_reason is not None:
            text(self.limit_override_reason, "limit_override_reason")
        if abs(self.acceleration_mps2) > 6 + 1e-6 or abs(self.curvature_per_m) > .3 + 1e-6:
            raise ContractError("control exceeds physical limits")


@dataclass(frozen=True)
class ExecutionFeedback:
    requested_id: str | None
    active_command_id: int | None
    accepted: bool
    stage: Stage
    execute_authorized: bool
    proposal_valid: bool
    source: Source
    reason: str | None
    reconsider_event: bool
    progress: float | None
    completed: bool

    def __post_init__(self):
        _enum(self.stage, Stage)
        _enum(self.source, Source)
        for name in ("accepted", "execute_authorized", "proposal_valid", "reconsider_event", "completed"):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"{name} must be bool")
        if self.active_command_id is not None:
            integer(self.active_command_id, "active_command_id", 1)
        for name in ("requested_id", "reason"):
            if getattr(self, name) is not None:
                text(getattr(self, name), name)
        if self.progress is not None:
            finite(self.progress, "progress")


def future_times(tick):
    integer(tick, "tick")
    return (tick + np.arange(1, FUTURE_POINTS + 1, dtype=np.float64)) * DT


def rotation(yaw):
    finite(yaw, "yaw")
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def decode_waypoints(raw_u, position, yaw, velocity, acceleration):
    u = array(raw_u, (30, 2), "raw_u")
    p = array(position, (2,), "position")
    v = array(velocity, (2,), "velocity")
    a = array(acceleration, (2,), "acceleration")
    r = rotation(yaw)
    tau = future_times(0)[:, None]
    local = (v @ r) * tau + .5 * (a @ r) * tau**2 + 3 * tau**2 * u
    result = p + local @ r.T
    if not np.isfinite(result).all():
        raise ContractError("nonfinite decoded path")
    return result


def encode_waypoints(xy_world, position, yaw, velocity, acceleration):
    xy = array(xy_world, (30, 2), "xy_world")
    p = array(position, (2,), "position")
    v = array(velocity, (2,), "velocity")
    a = array(acceleration, (2,), "acceleration")
    r = rotation(yaw)
    tau = future_times(0)[:, None]
    return ((xy - p) @ r - (v @ r) * tau - .5 * (a @ r) * tau**2) / (3 * tau**2)


def _optional(annotation):
    return get_origin(annotation) is UnionType and type(None) in get_args(annotation)


def to_wire(record):
    """Named enums and explicit optional validity; never serialize slot IDs as features."""
    if isinstance(record, Enum):
        return record.value
    if isinstance(record, np.ndarray):
        return record.tolist()
    if is_dataclass(record):
        hints = get_type_hints(type(record))
        result = {}
        for f in fields(record):
            value = getattr(record, f.name)
            encoded = to_wire(value)
            result[f.name] = {"value": encoded, "valid": value is not None} if _optional(hints[f.name]) else encoded
        return result
    if isinstance(record, np.generic):
        return record.item()
    return record


def from_wire(cls, payload):
    """Strict record decode. Unknown/missing fields and ambiguous optionals fail."""
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(cls)}:
        raise ContractError(f"incorrect fields for {cls.__name__}")
    values = {}
    for name, kind in get_type_hints(cls).items():
        value = payload[name]
        if _optional(kind):
            if not isinstance(value, dict) or set(value) != {"value", "valid"} or type(value["valid"]) is not bool:
                raise ContractError(f"invalid optional {name}")
            if value["valid"] != (value["value"] is not None):
                raise ContractError(f"optional value/valid disagree: {name}")
            value = value["value"]
            kind = next(k for k in get_args(kind) if k is not type(None))
            if value is None:
                values[name] = None
                continue
        try:
            if isinstance(kind, type) and issubclass(kind, Enum):
                value = kind(value)
            elif is_dataclass(kind):
                value = from_wire(kind, value)
            elif kind is np.ndarray:
                value = np.asarray(value, dtype=np.float64)
            elif kind is bool and type(value) is not bool:
                raise ContractError(f"{name} must be bool")
            elif kind is int:
                integer(value, name)
            elif kind is float:
                finite(value, name)
            elif kind is str:
                text(value, name)
        except (ValueError, TypeError) as error:
            raise ContractError(f"{name}: {error}") from error
        values[name] = value
    return cls(**values)
