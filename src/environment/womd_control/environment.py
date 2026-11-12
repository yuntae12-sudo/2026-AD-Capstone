"""Single-scene KEEP/FOLLOW control foundation, without a training collector.

The SDC uses control dynamics; all other tracks follow the source log. Policies
only receive immutable current/past observations. Unsupported modes fail closed.
"""

from dataclasses import dataclass, replace
import uuid

import numpy as np

from src.contracts.observation import CoreObservation
from src.contracts.planning import (
    BehaviorCommand,
    CommandDraft,
    CommandRequest,
    ContractError,
    ControlCommand,
    Direction,
    DT,
    ExecutionFeedback,
    ExecutionRequest,
    Maneuver,
    ObservationToken,
    Receipt,
    RejectionReason,
    SCHEMA_VERSION,
    Source,
    Stage,
    TaskKind,
    TaskSpec,
    TerminalReason,
    WaypointProposal,
    array,
    encode_waypoints,
    future_times,
    to_wire,
)
from src.scenarios.womd20 import read_scenario, scenario_to_waymax, validate_scenario
from .control import (
    CheckResult,
    Ego,
    PDMemory,
    Trajectory,
    advance,
    check_trajectory,
    emergency,
    pd_control,
    TrackingError,
    tracking_error,
)
from .geometry import ResetRejected, box, build_route, overlaps, project, wrap
from .observation import build_observation


@dataclass(frozen=True)
class Boundary:
    token: ObservationToken
    needs_ppo_action: bool
    needs_sac_action: bool
    sac_observation: CoreObservation
    feedback: ExecutionFeedback
    terminated: bool
    truncated: bool
    terminal_reason: TerminalReason | None
    ppo_observation: None = None  # this milestone uses an explicit fixed provider


@dataclass(frozen=True)
class ExecutionAttempt:
    source: Source
    proposal_id: str
    generated_tick: int
    check: CheckResult


@dataclass(frozen=True)
class StepResult:
    pre_token: ObservationToken
    post_tick: int
    dt_exec: float
    executed_command_snapshot: BehaviorCommand
    raw_proposal: WaypointProposal | None
    check_result: CheckResult
    source: Source
    actual_control: ControlCommand
    terminated: bool
    truncated: bool
    terminal_reason: TerminalReason | None
    sample_valid: bool
    next_boundary: Boundary
    attempts: tuple[ExecutionAttempt, ...] = ()
    tracking_error: TrackingError | None = None
    # No training reward/transition is invented before reward/collector gates.


class SACPlanningEnvironment:
    """Development API, initially fixed KEEP/FOLLOW providers only.

    reset accepts a Scenario proto or (shard_path, record_index). No global JAX
    precision/device setting is changed by importing this module.
    """

    def __init__(self):
        self._initialized = False
        self._step_cache = {}

    @property
    def frame(self):
        return self.start_frame + self.tick

    @property
    def token(self):
        return ObservationToken(
            self.episode_uid, self._scenario.scenario_id, self.tick, self.revision
        )

    @property
    def ego(self):
        return self.ego_at(self.frame)

    def ego_at(self, frame):
        trajectory = self._state.sim_trajectory
        i = self.ego_index
        v = np.array(
            [trajectory.vel_x[i, frame], trajectory.vel_y[i, frame]], np.float64
        )
        valid = frame > 0 and bool(trajectory.valid[i, frame - 1])
        before = (
            np.array(
                [trajectory.vel_x[i, frame - 1], trajectory.vel_y[i, frame - 1]],
                np.float64,
            )
            if valid
            else v
        )
        return Ego(
            np.array([trajectory.x[i, frame], trajectory.y[i, frame]], np.float64),
            v,
            (v - before) / DT if valid else np.zeros(2),
            bool(valid),
            float(trajectory.yaw[i, frame]),
            float(trajectory.length[i, frame]),
            float(trajectory.width[i, frame]),
        )

    def _objects(self):
        trajectory = self._state.sim_trajectory
        result = {}
        for i in range(trajectory.num_objects):
            if i == self.ego_index or not bool(trajectory.valid[i, self.frame]):
                continue
            values = np.array(
                [
                    trajectory.x[i, self.frame],
                    trajectory.y[i, self.frame],
                    trajectory.vel_x[i, self.frame],
                    trajectory.vel_y[i, self.frame],
                    trajectory.yaw[i, self.frame],
                    trajectory.length[i, self.frame],
                    trajectory.width[i, self.frame],
                ],
                float,
            )
            if not np.isfinite(values).all() or np.any(values[-2:] <= 0):
                raise ContractError("DATA_CORRUPTION: invalid current object geometry")
            result[int(self._state.object_metadata.ids[i])] = Ego(
                values[:2],
                values[2:4],
                np.zeros(2),
                False,
                float(values[4]),
                float(values[5]),
                float(values[6]),
            )
        return result

    def reset(self, scene_key, seed=0, provider_spec="fixed_keep"):
        import jax
        import jax.numpy as jnp
        from waymax import config, datatypes, dynamics, env
        from waymax.metrics import roadgraph, overlap

        self._initialized = False
        if provider_spec not in ("fixed_keep", "fixed_follow"):
            raise ResetRejected("UNSUPPORTED_PROVIDER")
        source_scenario = (
            read_scenario(*scene_key) if isinstance(scene_key, tuple) else scene_key
        )
        scenario = type(source_scenario)()
        scenario.CopyFrom(source_scenario)
        times = validate_scenario(scenario)
        if np.any(np.abs(np.diff(times) - DT) > 0.005):
            raise ResetRejected("SOURCE_TIMESTEP_OUTSIDE_TOLERANCE")
        self._scenario = scenario
        self.ego_index = int(scenario.sdc_track_index)
        ego_valid = np.array([s.valid for s in scenario.tracks[self.ego_index].states])
        starts = [
            i
            for i in range(max(2, scenario.current_time_index), len(times) - 50)
            if ego_valid[i - 2 : i + 1].all()
        ]
        if not starts:
            raise ResetRejected("INSUFFICIENT_HISTORY_OR_REMAINING_TIME")
        self.start_frame = starts[0]
        self.last_frame = len(times) - 1
        if not ego_valid[self.start_frame :].all():
            raise ResetRejected("EGO_LOG_VALIDITY_GAP")
        self.tick = self.revision = self.counter = 0
        self.episode_uid = str(uuid.uuid4())
        self.ledger = {}
        self.old_plan = None
        self.failure_streak = 0
        self.goal_satisfied = False
        self.terminated = self.truncated = False
        self.terminal_reason = None
        self.target_cache = None
        self.target_last_tick = None
        self.target_estimated = False
        self.provider_spec = provider_spec
        self.source_timestamp = float(times[self.start_frame])
        self.source_clock_drift_s = 0.0
        state, self.conversion_report = scenario_to_waymax(scenario)
        key = (state.log_trajectory.num_objects, self.start_frame)
        if key not in self._step_cache:
            runtime = env.PlanningAgentEnvironment(
                dynamics.InvertibleBicycleModel(
                    dt=DT, max_accel=6.0, max_steering=0.3, normalize_actions=False
                ),
                config.EnvironmentConfig(
                    max_num_objects=state.log_trajectory.num_objects,
                    init_steps=self.start_frame + 1,
                    controlled_object=config.ObjectType.SDC,
                    compute_reward=False,
                ),
            )
            self._step_cache[key] = (runtime, jax.jit(runtime.step))
        self._runtime, self._step = self._step_cache[key]
        self._state = self._runtime.reset(state, jax.random.PRNGKey(seed))
        self._offroad = jax.jit(roadgraph.is_offroad)
        self._overlap = jax.jit(overlap.OverlapMetric().compute_overlap)
        self._datatypes, self._jnp = datatypes, jnp
        ego = self.ego
        if (
            not 0 <= ego.speed <= 15
            or min(ego.length, ego.width) <= 0
            or np.linalg.norm(ego.acceleration) > 6
        ):
            raise ResetRejected("INITIAL_STATE_OUTSIDE_ODD")
        self.route = build_route(scenario, ego, self.frame)
        if ego.speed > min(15, self.route.speed_limit(ego.position)) + 1e-6:
            raise ResetRejected("INITIAL_SPEED_LIMIT")
        if len(self.route.lane_ids) > 16:
            raise ResetRejected("REQUIRED_LANE_SLOT_OVERFLOW")
        if not np.any(
            np.isin(np.asarray(state.roadgraph_points.types), [15, 16])
            & np.asarray(state.roadgraph_points.valid)
        ):
            raise ResetRejected("MISSING_ROAD_EDGES")
        h = (self.last_frame - self.start_frame) * DT
        ta = min(h, max(0.0, (15 - ego.speed) / 6))
        max_distance = ego.speed * ta + 3 * ta**2 + 15 * (h - ta)
        from .geometry import sample

        if (
            np.linalg.norm(sample(self.route.center, self.route.goal_s) - ego.position)
            > max_distance
        ):
            raise ResetRejected("RESET_UNREACHABLE_GOAL")
        self.task = TaskSpec(
            TaskKind.ROUTE, self.route.route_id + ":goal", self.route.goal_s
        )
        previous = self.ego_at(self.frame - 1)
        average_speed = (ego.speed + previous.speed) / 2
        initial_a = np.clip((ego.speed - previous.speed) / DT, -6, 6)
        initial_k = (
            np.clip(wrap(ego.yaw - previous.yaw) / (average_speed * DT), -0.3, 0.3)
            if average_speed >= 0.2
            else 0.0
        )
        self.memory = PDMemory(last_a=float(initial_a), last_kappa=float(initial_k))
        if self._actual_failure() is not None:
            raise ResetRejected("INITIAL_COLLISION_OR_ROAD_VIOLATION")
        target = None
        if provider_spec == "fixed_follow":
            candidates = []
            ego_s = project(self.route.center, ego.position)[0]
            types = {int(track.id): int(track.object_type) for track in scenario.tracks}
            for object_id, other in self._objects().items():
                if types[object_id] != 1:
                    continue
                s, _, tangent = project(self.route.center, other.position)
                if (
                    s > ego_s
                    and self.route.contains(
                        other.position, other.yaw, other.length, other.width
                    )
                    and abs(wrap(other.yaw - np.arctan2(tangent[1], tangent[0])))
                    <= np.pi / 6
                ):
                    candidates.append(
                        (s - ego_s - (ego.length + other.length) / 2, object_id)
                    )
            if not candidates:
                raise ResetRejected("NO_FOLLOW_TARGET")
            target = min(candidates)[1]
        draft = CommandDraft(
            Maneuver.FOLLOW if target is not None else Maneuver.KEEP,
            self.route.route_id,
            self.route.lane_ids[0],
            None,
            target,
            None,
            Direction.AUTO,
            min(10.0, self.route.speed_limit(ego.position)),
            1.5 if target is not None else None,
            3.0 if target is not None else None,
            None,
            None,
            self.route.route_id + ":corridor",
            "fixed_binding_v1",
        )
        self.command = BehaviorCommand(
            1, draft, 0, 10, SCHEMA_VERSION, "fixed_provider_v1"
        )
        self.counter = 1
        self.feedback = ExecutionFeedback(
            None,
            1,
            True,
            Stage.EXECUTING,
            True,
            False,
            Source.NONE,
            None,
            False,
            None,
            False,
        )
        self._refresh_target()
        self._initialized = True
        return self._boundary()

    def _require(self, token):
        if not self._initialized:
            raise ContractError("reset required")
        token.require_current(self.token)
        if self.terminated or self.truncated:
            raise ContractError("episode already ended")

    def _boundary(self):
        observation = (
            CoreObservation.empty()
            if self.terminal_reason == TerminalReason.DATA_CORRUPTION
            else build_observation(self)
        )
        return Boundary(
            self.token,
            False,
            not (self.terminated or self.truncated),
            observation,
            self.feedback,
            self.terminated,
            self.truncated,
            self.terminal_reason,
        )

    def planning_anchor(self, token):
        self._require(token)
        ego = self.ego
        return replace(
            ego,
            position=array(ego.position, (2,), "position"),
            velocity=array(ego.velocity, (2,), "velocity"),
            acceleration=array(ego.acceleration, (2,), "acceleration"),
        )

    def proposal_from_world(
        self, xy_world, token, sequence=0, model_version="scripted-development"
    ):
        ego = self.planning_anchor(token)
        return WaypointProposal(
            token,
            self.command.command_id,
            f"{token.episode_uid}:{token.sim_tick}:{sequence}",
            token.sim_tick,
            ego.position,
            ego.yaw,
            ego.velocity,
            ego.acceleration,
            ego.acceleration_valid,
            future_times(token.sim_tick),
            xy_world,
            encode_waypoints(
                xy_world, ego.position, ego.yaw, ego.velocity, ego.acceleration
            ),
            model_version=model_version,
        )

    def submit_behavior(self, request: CommandRequest, token):
        if not isinstance(request, CommandRequest) or not isinstance(
            token, ObservationToken
        ):
            raise ContractError("CommandRequest and ObservationToken required")
        if not self._initialized or token.episode_uid != self.episode_uid:
            raise ContractError("STALE_OBSERVATION")
        fingerprint = to_wire(request)
        if request.request_id in self.ledger:
            previous, receipt = self.ledger[request.request_id]
            if previous != fingerprint:
                raise ContractError("DUPLICATE_REQUEST_MISMATCH")
            return receipt, self._boundary()
        self._require(token)
        draft = request.command_draft
        reason = None
        if request.candidate_token is not None:
            raise ContractError(
                "PPO candidate builder is not enabled by fixed provider"
            )
        if (
            draft.maneuver not in (Maneuver.KEEP, Maneuver.FOLLOW)
            or request.execution_request != ExecutionRequest.EXECUTE
        ):
            reason = RejectionReason.UNSUPPORTED_MODE
        elif (
            draft.route_id != self.route.route_id
            or draft.allowed_corridor_id != self.route.route_id + ":corridor"
            or draft.source_lane_id not in self.route.lane_ids
            or (
                draft.target_lane_id is not None
                and draft.target_lane_id not in self.route.lane_ids
            )
        ):
            reason = RejectionReason.INVALID_CORRIDOR
        elif (
            draft.maneuver == Maneuver.FOLLOW
            and draft.target_agent_id not in self._objects()
        ):
            reason = RejectionReason.MISSING_TARGET
        elif not self.command.is_valid_at(self.tick):
            reason = RejectionReason.EXPIRED_COMMAND
        self.revision += 1
        if reason is None:
            if to_wire(draft) == to_wire(self.command.draft):
                self.command = replace(self.command, valid_until_tick=self.tick + 10)
            else:
                self.counter += 1
                self.command = BehaviorCommand(
                    self.counter,
                    draft,
                    self.tick,
                    self.tick + 10,
                    SCHEMA_VERSION,
                    request.provider_version,
                )
                self.memory = replace(self.memory, errors_valid=False)
                self.old_plan = None
                self.failure_streak = 0
                self.target_cache = None
                self.target_last_tick = None
                self._refresh_target()
        receipt = Receipt(
            request.request_id,
            reason is None,
            reason,
            self.command.command_id,
            self.revision,
        )
        self.ledger[request.request_id] = (fingerprint, receipt)
        self.feedback = replace(
            self.feedback,
            requested_id=request.request_id,
            accepted=reason is None,
            active_command_id=self.command.command_id,
            reason=reason.value if reason else None,
        )
        return receipt, self._boundary()

    def _refresh_target(self):
        target = self.command.draft.target_agent_id
        self.target_estimated = False
        if target is None:
            return True
        objects = self._objects()
        if target in objects:
            self.target_cache = objects[target]
            self.target_last_tick = self.tick
            return True
        self.target_estimated = (
            self.target_last_tick is not None and self.tick - self.target_last_tick <= 2
        )
        return self.target_estimated

    def _check_objects(self):
        objects = self._objects()
        if self.target_estimated:
            target = self.target_cache
            objects[self.command.draft.target_agent_id] = replace(
                target,
                position=target.position
                + target.velocity * (self.tick - self.target_last_tick) * DT,
            )
        return list(objects.values())

    def _actual_failure(self):
        import jax

        ego = self.ego
        if (
            not np.isfinite(
                np.r_[ego.position, ego.velocity, ego.yaw, ego.length, ego.width]
            ).all()
            or min(ego.length, ego.width) <= 0
        ):
            self.waymax_offroad = self.waymax_overlap = None
            self.metrics_valid = False
            return TerminalReason.DATA_CORRUPTION
        rectangle = box(ego.position, ego.yaw, ego.length, ego.width)
        try:
            objects = self._objects()
        except ContractError:
            self.waymax_offroad = self.waymax_overlap = None
            self.metrics_valid = False
            return TerminalReason.DATA_CORRUPTION
        collision = False
        for other in objects.values():
            if overlaps(
                rectangle, box(other.position, other.yaw, other.length, other.width)
            ):
                collision = True
        trajectory = jax.tree_util.tree_map(
            lambda x: x[
                self.ego_index : self.ego_index + 1, self.frame : self.frame + 1
            ],
            self._state.sim_trajectory,
        )
        self.waymax_offroad = bool(
            np.asarray(self._offroad(trajectory, self._state.roadgraph_points))[0]
        )
        all_current = jax.tree_util.tree_map(
            lambda x: x[:, self.frame : self.frame + 1], self._state.sim_trajectory
        )
        metric = self._overlap(all_current)
        self.waymax_overlap = float(np.asarray(metric.value)[self.ego_index])
        self.metrics_valid = bool(np.asarray(metric.valid)[self.ego_index])
        if not self.metrics_valid:
            return TerminalReason.DATA_CORRUPTION
        if collision:
            return TerminalReason.COLLISION
        if self.waymax_offroad:
            return TerminalReason.OFFROAD
        if not self.route.contains(ego.position, ego.yaw, ego.length, ego.width):
            return TerminalReason.CORRIDOR_VIOLATION
        return None

    def step_waypoints(self, proposal, token):
        if not isinstance(proposal, WaypointProposal) or not isinstance(
            token, ObservationToken
        ):
            raise ContractError("WaypointProposal and ObservationToken required")
        self._require(token)
        proposal.observation_token.require_current(self.token)
        if (
            proposal.command_id != self.command.command_id
            or proposal.generated_tick != self.tick
        ):
            raise ContractError("command/tick mismatch")
        ego = self.ego
        if (
            np.max(np.abs(proposal.anchor_world_xy - ego.position)) > 1e-7
            or np.max(np.abs(proposal.anchor_velocity_world - ego.velocity)) > 1e-7
            or np.max(np.abs(proposal.anchor_acceleration_world - ego.acceleration))
            > 1e-7
            or abs(wrap(proposal.anchor_yaw - ego.yaw)) > 1e-7
            or proposal.anchor_acceleration_valid != ego.acceleration_valid
        ):
            raise ContractError("ANCHOR_MISMATCH")
        if not self.command.is_valid_at(self.tick):
            raise ContractError("EXPIRED_COMMAND")
        trajectory = Trajectory(proposal)
        objects = self._check_objects()
        check = self._check(trajectory, objects)
        attempts = [
            ExecutionAttempt(
                Source.NEW, proposal.proposal_id, proposal.generated_tick, check
            )
        ]
        chosen, source = None, Source.EMERGENCY
        if check.valid:
            chosen, source = trajectory, Source.NEW
            self.old_plan = trajectory
            self.failure_streak = 0
        else:
            self.failure_streak += 1
            if (
                self.old_plan is not None
                and self.old_plan.proposal.command_id == self.command.command_id
            ):
                if self.tick - self.old_plan.proposal.generated_tick <= 2:
                    reused = self._check(self.old_plan, objects)
                    if reused.valid:
                        chosen, source = self.old_plan, Source.REUSED
                else:
                    reused = CheckResult(False, "OLD_PLAN_EXPIRED")
                attempts.append(
                    ExecutionAttempt(
                        Source.REUSED,
                        self.old_plan.proposal.proposal_id,
                        self.old_plan.proposal.generated_tick,
                        reused,
                    )
                )
                if chosen is None:
                    braking = self._braking_trajectory()
                    brake_check = self._check(braking, objects)
                    attempts.append(
                        ExecutionAttempt(
                            Source.BRAKE,
                            braking.proposal.proposal_id,
                            braking.proposal.generated_tick,
                            brake_check,
                        )
                    )
                    if brake_check.valid:
                        chosen, source = braking, Source.BRAKE
        if chosen is None:
            control, new_memory = emergency(ego, self.memory, self.tick)
        else:
            control, new_memory = pd_control(
                chosen, ego, self.memory, self.tick, source
            )
        return self._advance(
            proposal, check, control, new_memory, chosen, tuple(attempts)
        )

    def _check(self, trajectory, objects):
        ego = self.ego
        check = check_trajectory(
            trajectory, ego, self.route, objects, self.memory, self.tick
        )
        # FOLLOW must respect its original target; the target is never rebound.
        if check.valid and self.command.draft.maneuver == Maneuver.FOLLOW:
            target = self._objects().get(self.command.draft.target_agent_id)
            if target is None and self.target_estimated:
                target = replace(
                    self.target_cache,
                    position=self.target_cache.position
                    + self.target_cache.velocity
                    * (self.tick - self.target_last_tick)
                    * DT,
                )
            if target is None:
                check = CheckResult(False, "MISSING_TARGET")
            else:
                for tau in np.arange(0, trajectory.end - self.tick * DT + 1e-8, 0.1):
                    position = trajectory.evaluate(self.tick * DT + tau)
                    s = project(self.route.center, position)[0]
                    target_s = project(
                        self.route.center, target.position + target.velocity * tau
                    )[0]
                    speed = np.linalg.norm(trajectory.evaluate(self.tick * DT + tau, 1))
                    if (
                        target_s - s - (ego.length + target.length) / 2
                        < self.command.draft.minimum_gap_m
                        + self.command.draft.time_gap_s * speed
                    ):
                        check = CheckResult(False, "COMMAND_VIOLATION")
                        break
        return check

    def _braking_trajectory(self):
        # New candidate from current actual state. The old plan's age is never
        # refreshed; beyond its end only the last valid curvature is retained.
        virtual = self.ego
        last_a, last_kappa = self.memory.last_a, self.memory.last_kappa
        positions = []
        for step in range(30):
            time = min((self.tick + step) * DT, self.old_plan.end)
            velocity, acceleration = [self.old_plan.evaluate(time, d) for d in (1, 2)]
            speed = np.linalg.norm(velocity)
            if speed >= 0.01:
                desired_k = (
                    velocity[0] * acceleration[1] - velocity[1] * acceleration[0]
                ) / speed**3
            else:
                desired_k = last_kappa
            last_a = float(
                np.clip(
                    # A 20 m/s^3 step in generator controls overshoots that
                    # limit after finite differences + quintic reconstruction.
                    # Reserve interpolation headroom (1.5 m/s^3 here); interior
                    # quintic jerk can reach 9 times the generator ramp. The shared
                    # checker still enforces the original 20 m/s^3 limit.
                    np.clip(-2.0, last_a - 0.15, last_a + 0.15),
                    max(-6.0, -virtual.speed / DT),
                    6.0,
                )
            )
            last_kappa = float(
                np.clip(
                    np.clip(desired_k, last_kappa - 0.05, last_kappa + 0.05), -0.3, 0.3
                )
            )
            virtual = advance(virtual, last_a, last_kappa)
            positions.append(virtual.position)
        proposal = self.proposal_from_world(
            np.array(positions),
            self.token,
            sequence="brake",
            model_version="checked-brake-v2",
        )
        return Trajectory(proposal)

    def step_forced(self, token):
        self._require(token)
        # No HOLD or forced lifecycle is activated in the initial KEEP/FOLLOW API.
        raise ContractError("needs_sac_action is true; forced tick is unavailable")

    def _advance(
        self,
        proposal,
        check,
        control,
        new_memory,
        executed_trajectory=None,
        attempts=(),
    ):
        before = self.token
        executed = self.command
        action = self._datatypes.Action(
            data=self._jnp.asarray(
                [control.acceleration_mps2, control.curvature_per_m],
                dtype=self._jnp.float32,
            ),
            valid=self._jnp.asarray([True]),
        )
        self._state = self._step(self._state, action)
        # Waymax's dynamics update carries dimensions from the previous sim
        # frame. A newly appearing logged object can therefore be valid with
        # zero dimensions. Restore exogenous size/time at the *new* source
        # frame before geometry checks; never overwrite controlled ego XY/v/yaw.
        next_frame = self.frame + 1
        sim, log = self._state.sim_trajectory, self._state.log_trajectory
        updates = {
            name: getattr(sim, name)
            .at[:, next_frame]
            .set(getattr(log, name)[:, next_frame])
            for name in ("length", "width", "height", "timestamp_micros")
        }
        z = (
            log.z[:, next_frame]
            .at[self.ego_index]
            .set(sim.z[self.ego_index, next_frame])
        )
        updates["z"] = sim.z.at[:, next_frame].set(z)
        self._state = self._state.replace(sim_trajectory=sim.replace(**updates))
        self.tick += 1
        self.revision = 0
        self.memory = new_memory
        self.source_timestamp = float(self._scenario.timestamps_seconds[self.frame])
        self.source_clock_drift_s = (
            self.source_timestamp
            - float(self._scenario.timestamps_seconds[self.start_frame])
            - self.tick * DT
        )
        failure = self._actual_failure()
        target_valid = (
            self._refresh_target()
            if failure != TerminalReason.DATA_CORRUPTION
            else False
        )
        s = project(self.route.center, self.ego.position)[0]
        self.goal_satisfied = bool(s >= self.route.goal_s and failure is None)
        if failure is not None:
            self.terminated = True
            self.terminal_reason = failure
        elif not target_valid:
            self.terminated = True
            self.terminal_reason = TerminalReason.TASK_LOST
        elif self.goal_satisfied:
            self.terminated = True
            self.terminal_reason = TerminalReason.ROUTE_GOAL
        elif self.frame >= self.last_frame:
            self.truncated = True
            self.terminal_reason = TerminalReason.RECORD_END
        elif self.tick >= 200:
            self.truncated = True
            self.terminal_reason = TerminalReason.TIME_LIMIT
        if not (self.terminated or self.truncated) and self.tick % 5 == 0:
            self.command = replace(self.command, valid_until_tick=self.tick + 10)
        self.feedback = ExecutionFeedback(
            self.feedback.requested_id,
            self.command.command_id,
            True,
            (
                Stage.COMPLETED
                if self.goal_satisfied
                else Stage.FAILED if self.terminated else Stage.EXECUTING
            ),
            not (self.terminated or self.truncated),
            check.valid,
            control.source,
            check.reason if not check.valid else None,
            self.failure_streak == 3,
            float(s - self.route.start_s),
            self.goal_satisfied,
        )
        boundary = self._boundary()
        return (
            StepResult(
                before,
                self.tick,
                DT,
                executed,
                proposal,
                check,
                control.source,
                control,
                self.terminated,
                self.truncated,
                self.terminal_reason,
                self.terminal_reason != TerminalReason.DATA_CORRUPTION,
                boundary,
                attempts,
                tracking_error(executed_trajectory, self.ego, self.tick * DT),
            ),
            boundary,
        )


def scripted_proposal(environment, token):
    """Current-state kinematic development input; not an ASMC/learned planner.

    Continue measured velocity/acceleration to exercise the common checker.
    Rejection is expected where this extension is not physically feasible.
    """
    from src.contracts.planning import decode_waypoints

    ego = environment.planning_anchor(token)
    xy = decode_waypoints(
        np.zeros((30, 2)), ego.position, ego.yaw, ego.velocity, ego.acceleration
    )
    return environment.proposal_from_world(
        xy, token, model_version="constant-acceleration-development-v1"
    )


def route_reference_proposal(environment, token):
    """Deterministic current-map quintic reference for control diagnostics.

    This is a separate development input, not an ASMC implementation or learned
    policy. It reads no future logs, performs no candidate search, and receives
    exactly the same acceptance checks as every other proposal.
    """
    from .geometry import sample

    e = environment
    ego = e.planning_anchor(token)
    route_s, _, _ = project(e.route.center, ego.position)
    desired = min(
        e.command.draft.desired_speed_mps, e.route.speed_limit(ego.position), 15.0
    )
    if e.command.draft.maneuver == Maneuver.FOLLOW and e.target_cache is not None:
        target = e.target_cache
        target_position = (
            target.position + target.velocity * (e.tick - e.target_last_tick) * DT
        )
        gap = (
            project(e.route.center, target_position)[0]
            - route_s
            - (ego.length + target.length) / 2
        )
        desired = max(
            0.0,
            min(
                desired,
                target.speed
                + 0.5
                * (
                    gap
                    - e.command.draft.minimum_gap_m
                    - e.command.draft.time_gap_s * ego.speed
                ),
            ),
        )
    duration = 3.0
    target_s = route_s + 0.5 * (ego.speed + desired) * duration
    if target_s > e.route.length:
        raise ContractError("development reference exceeds available route")
    target_position = sample(e.route.center, target_s)
    target_tangent = project(e.route.center, target_position)[2]
    # Polynomial in normalized time, calculated relative to the actual anchor
    # to avoid cancellation at large World coordinates.
    coefficients = np.zeros((2, 6), np.float64)
    coefficients[:, 1] = ego.velocity * duration
    coefficients[:, 2] = ego.acceleration * duration**2 / 2
    rhs = np.stack(
        [
            target_position - ego.position - coefficients[:, 1] - coefficients[:, 2],
            desired * target_tangent * duration
            - coefficients[:, 1]
            - 2 * coefficients[:, 2],
            -2 * coefficients[:, 2],
        ],
        axis=-1,
    )
    inverse = np.linalg.inv(np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], float))
    coefficients[:, 3:] = rhs @ inverse.T
    z = np.arange(1, 31) * DT / duration
    xy = ego.position + np.sum(
        coefficients[None, :, :] * z[:, None, None] ** np.arange(6), axis=-1
    )
    return e.proposal_from_world(
        xy, token, model_version="current-route-quintic-development-v1"
    )
