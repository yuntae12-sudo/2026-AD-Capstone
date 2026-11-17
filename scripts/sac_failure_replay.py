"""Verify saved actions against CPU Waymax before attributing collision contacts."""

from collections import deque
from dataclasses import replace
import numpy as np


def relative_state(ego, other):
    h = np.array([np.cos(ego.yaw), np.sin(ego.yaw)])
    left = np.array([-h[1], h[0]])
    offset = other.position - ego.position
    relative_velocity = other.velocity - ego.velocity
    longitudinal = float(offset @ h)
    longitudinal_velocity = float(relative_velocity @ h)
    return dict(
        ego_speed=ego.speed,
        other_speed=other.speed,
        relative_longitudinal_m=longitudinal,
        relative_lateral_m=float(offset @ left),
        relative_longitudinal_velocity_mps=longitudinal_velocity,
        closing_speed_mps=float(-np.sign(longitudinal) * longitudinal_velocity),
        center_side=(
            "FRONT"
            if longitudinal > 0.5
            else "REAR" if longitudinal < -0.5 else "ALONGSIDE"
        ),
    )


def replay_episode(rows, scenario, seed):
    from src.contracts.planning import WaypointProposal, from_wire, to_wire
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.environment.womd_control.geometry import box, overlaps, wrap
    from src.training.sac.collector import collect_step

    e = SACPlanningEnvironment()
    b = e.reset(scenario, seed=seed, provider_spec=rows[0]["provider"])
    history = deque(maxlen=31)
    history.append((0, e.ego, e._objects()))
    verified_post_anchors = 0
    reasons = {}
    for index, row in enumerate(rows):
        original = from_wire(WaypointProposal, row["raw_proposal"])
        proposal = replace(
            original,
            observation_token=b.token,
            command_id=e.command.command_id,
            proposal_id=f"failure-replay-{e.tick}",
        )
        step = collect_step(e, b, lambda *_: proposal)
        result, b = step.result, step.boundary
        if (
            result.source.value != row["source"]
            or result.check_result.valid != row["check_valid"]
            or result.check_result.reason != row["check_reason"]
            or list(result.event_flags) != row["event_flags"]
            or result.terminated != row["terminated"]
            or result.truncated != row["truncated"]
            or result.sample_valid != row["sample_valid"]
            or (result.terminal_reason.value if result.terminal_reason else None)
            != row["terminal_reason"]
            or to_wire(result.actual_control) != row["actual_control"]
        ):
            raise ValueError(
                f"replay control/check/event divergence at tick {row['tick']}"
            )
        np.testing.assert_allclose(
            step.reward.training, row["reward"], rtol=0, atol=1e-9
        )
        if index + 1 < len(rows):
            nxt = rows[index + 1]["raw_proposal"]
            for actual, expected in (
                (e.ego.position, nxt["anchor_world_xy"]),
                (e.ego.velocity, nxt["anchor_velocity_world"]),
                (e.ego.acceleration, nxt["anchor_acceleration_world"]),
            ):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
            if abs(wrap(e.ego.yaw - nxt["anchor_yaw"])) > 1e-7:
                raise ValueError("replay yaw divergence")
            verified_post_anchors += 1
        reasons[str(row["tick"])] = list(result.check_result.violations)
        history.append((e.tick, e.ego, e._objects()))
    rect = box(e.ego.position, e.ego.yaw, e.ego.length, e.ego.width)
    contacts = []
    for object_id, other in e._objects().items():
        if not overlaps(
            rect, box(other.position, other.yaw, other.length, other.width)
        ):
            continue
        track = [
            dict(tick=t, time_s=t * 0.1, **relative_state(ego, objects[object_id]))
            for t, ego, objects in history
            if object_id in objects
        ]
        approach = next(
            (r for r in track[:-1] if r["center_side"] != "ALONGSIDE"), track[-1]
        )
        contacts.append(
            dict(
                object_id=object_id,
                approach_side=approach["center_side"],
                final=track[-1],
                last_3s=track,
            )
        )
    return dict(
        episode_uid=rows[0]["episode_uid"],
        scene_id=rows[0]["scene_id"],
        provider=rows[0]["provider"],
        status="VERIFIED",
        verified_ticks=len(rows),
        verified_post_anchors=verified_post_anchors,
        final_state_evidence="reproduced from verified controls; terminal post-state absent from original trace",
        final_ego_speed=e.ego.speed,
        contacts=contacts,
        last_3s_kinematic_violations={
            str(t): reasons[str(t)] for t, _, _ in history if str(t) in reasons
        },
        chronological_pattern=(
            "BRAKING_THEN_STOPPED_REAR_CONTACT"
            if e.ego.speed < 0.1
            and any(c["approach_side"] == "REAR" for c in contacts)
            and any(r["source"] in ("BRAKE", "EMERGENCY") for r in rows[-30:])
            else (
                "CONTACT_IDENTIFIED"
                if contacts
                else "COLLISION_WITHOUT_OBB_CONTACT_IDENTIFIED"
            )
        ),
        limitation="Relative centers describe approach geometry, not legal fault or inevitability. Logged actors do not react to ego braking.",
    )
