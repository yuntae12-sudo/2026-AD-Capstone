#!/usr/bin/env python3
"""Replay saved proposals exactly for collision attribution; never feeds training."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    import numpy as np
    from src.contracts.planning import WaypointProposal, from_wire
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.environment.womd_control.geometry import box, overlaps
    from src.scenarios.womd20 import parse_scenario
    from src.training.sac.runtime import selected_scenes, versions, write_report

    version = versions(args.manifest)
    meta = json.loads((args.collection / "report.json").read_text())
    if meta["versions"] != version:
        raise ValueError("replay must use the exact collection version")
    candidates = {
        (r["scene_id"], r["provider"]): p
        for r, p in selected_scenes(args.manifest, "train")
    }
    selected = []
    pending = []
    with (args.collection / "ticks.jsonl").open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if pending and row["episode_uid"] != pending[-1]["episode_uid"]:
                pending = []
            pending.append(row)
            if row["terminated"] or row["truncated"]:
                if row["terminal_reason"] == "COLLISION":
                    selected.append(pending)
                pending = []
                if len(selected) == args.episodes:
                    break
    report = dict(
        scope="SAVED_ACTION_COLLISION_REPLAY_DIAGNOSTIC", versions=version, episodes=[]
    )
    for rows in selected:
        first = rows[0]
        e = SACPlanningEnvironment()
        b = e.reset(
            parse_scenario(
                candidates[(first["scene_id"], first["provider"])].read_bytes()
            ),
            seed=meta["seed"],
            provider_spec=first["provider"],
        )
        history = []
        for row in rows:
            original = from_wire(WaypointProposal, row["raw_proposal"])
            proposal = replace(
                original,
                observation_token=b.token,
                command_id=e.command.command_id,
                proposal_id=f"diagnostic-{e.tick}",
            )
            result, b = e.step_waypoints(proposal, b.token)
            if (
                result.source.value != row["source"]
                or result.check_result.reason != row["check"]["reason"]
            ):
                raise AssertionError("replay control/check divergence")
            np.testing.assert_allclose(
                e.ego.position, row["actual_state"]["world_xy"], atol=1e-8, rtol=0
            )
            np.testing.assert_allclose(
                e.ego.velocity, row["actual_state"]["velocity_world"], atol=1e-8, rtol=0
            )
            if tuple(result.event_flags) != tuple(row["event_flags"]):
                raise AssertionError("replay event divergence")
            history.append((e.tick, e.ego, e._objects()))
        rectangle = box(e.ego.position, e.ego.yaw, e.ego.length, e.ego.width)
        collided = [
            object_id
            for object_id, other in e._objects().items()
            if overlaps(
                rectangle, box(other.position, other.yaw, other.length, other.width)
            )
        ]
        collisions = []
        for object_id in collided:
            track = []
            for tick, ego, objects in history[-31:]:
                if object_id not in objects:
                    continue
                other = objects[object_id]
                h = np.array([np.cos(ego.yaw), np.sin(ego.yaw)])
                offset = other.position - ego.position
                track.append(
                    dict(
                        tick=tick,
                        ego_speed=ego.speed,
                        other_speed=other.speed,
                        relative_longitudinal_m=float(offset @ h),
                        relative_lateral_m=float(offset @ np.array([-h[1], h[0]])),
                    )
                )
            collisions.append(dict(object_id=object_id, last_3s=track))
        episode = dict(
            scene_id=first["scene_id"],
            provider=first["provider"],
            ticks=len(rows),
            exact_state_control_event_replay=True,
            collisions=collisions,
            first_rejection=next(
                (
                    dict(tick=r["tick"], check=r["check"], source=r["source"])
                    for r in rows
                    if not r["check"]["valid"]
                ),
                None,
            ),
            final_ego_speed=e.ego.speed,
        )
        report["episodes"].append(episode)
        write_report(args.output, report)
        print(
            json.dumps({k: v for k, v in episode.items() if k != "collisions"}),
            flush=True,
        )


if __name__ == "__main__":
    main()
