"""Read-only visualization capture of actual Waymax control rollouts."""

from collections import Counter
import numpy as np

from src.environment.womd_control.environment import SACPlanningEnvironment
from src.environment.womd_control.geometry import box, overlaps
from src.training.sac.collector import collect_step


class CapturingEnvironment(SACPlanningEnvironment):
    """Observe the selected trajectory without replacing any control/check logic."""

    def _advance(
        self,
        proposal,
        check,
        control,
        new_memory,
        executed_trajectory=None,
        attempts=(),
    ):
        self.visual_executed_xy = []
        if executed_trajectory is not None:
            times = np.arange(self.tick + 1, self.tick + 31) * 0.1
            times = times[times <= executed_trajectory.end + 1e-9]
            self.visual_executed_xy = np.asarray(
                executed_trajectory.evaluate(times)
            ).tolist()
        return super()._advance(
            proposal, check, control, new_memory, executed_trajectory, attempts
        )


def vehicle(ego, object_id=None):
    return dict(
        id=object_id,
        xy=ego.position.tolist(),
        yaw=ego.yaw,
        length=ego.length,
        width=ego.width,
        speed=ego.speed,
        velocity=ego.velocity.tolist(),
    )


def snapshot(e, result=None, reward=None):
    ego, objects = e.ego, e._objects()
    rectangle = box(ego.position, ego.yaw, ego.length, ego.width)
    frame = dict(
        tick=e.tick,
        time_s=e.tick * 0.1,
        ego=vehicle(ego),
        objects=[vehicle(v, k) for k, v in objects.items()],
        target_id=e.command.draft.target_agent_id,
        raw_xy=[],
        executed_xy=[],
        accepted=None,
        source="RESET",
        reason="RESET",
        violations=[],
        events=[],
        outcome=None,
        reward=0.0,
        acceleration=None,
        curvature=None,
        proposal_tick=None,
        colliding_ids=[
            k
            for k, v in objects.items()
            if overlaps(rectangle, box(v.position, v.yaw, v.length, v.width))
        ],
    )
    if result is not None:
        frame.update(
            raw_xy=result.raw_proposal.xy_world.tolist(),
            executed_xy=e.visual_executed_xy,
            proposal_tick=result.raw_proposal.generated_tick,
            accepted=bool(result.check_result.valid),
            source=result.source.value,
            reason=result.check_result.reason,
            violations=list(result.check_result.violations),
            events=list(result.event_flags),
            outcome=result.terminal_reason.value if result.terminal_reason else None,
            reward=float(reward.training),
            sample_valid=bool(result.sample_valid),
            acceleration=result.actual_control.acceleration_mps2,
            curvature=result.actual_control.curvature_per_m,
        )
    return frame


def map_snapshot(scenario, route):
    lines = []
    for feature in scenario.map_features:
        kind = feature.WhichOneof("feature_data")
        if kind in ("lane", "road_line", "road_edge"):
            xy = [[p.x, p.y] for p in getattr(feature, kind).polyline]
            if len(xy) > 1:
                lines.append(dict(kind=kind, xy=xy))
    return dict(
        lines=lines, route=route.center.tolist(), corridor=route.polygon.tolist()
    )


def capture(scenario, provider, proposer, label, seed=0, max_ticks=200):
    if max_ticks < 1:
        raise ValueError("max_ticks must be positive")
    e = CapturingEnvironment()
    b = e.reset(scenario, seed=seed, provider_spec=provider)
    frames = [snapshot(e)]
    mapping = map_snapshot(scenario, e.route)
    for _ in range(max_ticks):
        step = collect_step(e, b, proposer)  # No replay; no learner update.
        b = step.boundary
        frames.append(snapshot(e, step.result, step.reward))
        if step.result.terminated or step.result.truncated:
            break
    steps = frames[1:]
    return dict(
        label=label,
        scene_id=scenario.scenario_id,
        provider=provider,
        seed=seed,
        map=mapping,
        frames=frames,
        outcome=frames[-1]["outcome"] or "PREVIEW_LIMIT",
        summary=dict(
            ticks=len(steps),
            initial_speed=frames[0]["ego"]["speed"],
            rejected=sum(not f["accepted"] for f in steps),
            braking=sum(f["source"] in ("BRAKE", "EMERGENCY") for f in steps),
            stationary=sum(f["ego"]["speed"] < 0.1 for f in steps),
            return_training=sum(f["reward"] for f in steps),
            reasons=dict(Counter(f["reason"] for f in steps)),
        ),
    )


def require_paired(episodes):
    if len(episodes) != 2:
        raise ValueError("comparison requires exactly two episodes")
    a, b = episodes
    if (a["scene_id"], a["provider"], a["seed"]) != (
        b["scene_id"],
        b["provider"],
        b["seed"],
    ):
        raise ValueError("comparison scene/provider/seed mismatch")
    if a["frames"][0] != b["frames"][0] or a["map"] != b["map"]:
        raise ValueError("comparison initial states/map mismatch")
    for episode in episodes:
        if [f["tick"] for f in episode["frames"]] != list(
            range(len(episode["frames"]))
        ):
            raise ValueError("noncontiguous visualization ticks")


def frame_at(episode, tick):
    """Hold the final state with an explicit ended flag; never invent future motion."""
    index = min(tick, len(episode["frames"]) - 1)
    return episode["frames"][index], tick > index
