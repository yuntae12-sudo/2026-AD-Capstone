#!/usr/bin/env python3
"""Audit a consistent byte prefix of a live collection without altering it."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import math


def audit(path):
    path = Path(path)
    metadata_path = path.parent / "report.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    limit = path.stat().st_size
    digest = hashlib.sha256()
    counts = Counter()
    behavior = defaultdict(Counter)
    scenes = Counter()
    signatures = set()
    episodes = {}
    consumed = 0
    with path.open("rb") as stream:
        while consumed < limit:
            line = stream.readline(limit - consumed)
            consumed += len(line)
            if not line.endswith(b"\n"):
                break  # A live writer's incomplete last record is not corruption.
            digest.update(line)
            row = json.loads(line)
            key = row["episode_uid"]
            e = episodes.setdefault(
                key,
                dict(
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    ticks=0,
                    stationary_ticks=0,
                    last_tick=0,
                    terminal=None,
                    first_rejection=None,
                    first_emergency=None,
                    tail=[],
                ),
            )
            if row["tick"] != e["last_tick"] + 1:
                raise ValueError(f"noncontiguous tick in episode {key}")
            if e["terminal"] is not None:
                raise ValueError("trace continues after terminal")
            e["last_tick"] = row["tick"]
            e["ticks"] += 1
            group = row["provider"]
            counts["ticks"] += 1
            behavior[group]["ticks"] += 1
            scenes[row["scene_id"]] += 1
            counts["valid_transitions"] += int(row["sample_valid"])
            behavior[group]["valid_transitions"] += int(row["sample_valid"])
            accepted = row["sample_valid"] and row["check"]["valid"]
            input_stationary = (
                math.hypot(*row["raw_proposal"]["anchor_velocity_world"]) < 0.1
            )
            counts["bc_eligible"] += int(accepted)
            behavior[group]["bc_eligible"] += int(accepted)
            counts["bc_stationary"] += int(accepted and input_stationary)
            behavior[group]["bc_stationary"] += int(accepted and input_stationary)
            for label, value in (
                ("source", row["source"]),
                ("check", row["check"]["reason"]),
            ):
                counts[f"{label}:{value}"] += 1
                behavior[group][f"{label}:{value}"] += 1
            speed = None
            if row["actual_state"] is not None:
                speed = math.hypot(*row["actual_state"]["velocity_world"])
                if speed < 0.1:
                    counts["stationary_ticks"] += 1
                    behavior[group]["stationary_ticks"] += 1
                    e["stationary_ticks"] += 1
            # Exact action/state duplicates are reported, not silently removed.
            signature = json.dumps(
                [
                    row["scene_id"],
                    group,
                    row["tick"],
                    row["raw_proposal"]["raw_u"],
                    row["actual_state"],
                ],
                sort_keys=True,
            )
            h = hashlib.sha256(signature.encode()).digest()
            counts["duplicate_state_action_ticks"] += int(h in signatures)
            signatures.add(h)
            if not row["check"]["valid"] and e["first_rejection"] is None:
                e["first_rejection"] = dict(
                    tick=row["tick"],
                    reason=row["check"]["reason"],
                    violations=row["check"]["violations"],
                )
            if row["source"] == "EMERGENCY" and e["first_emergency"] is None:
                e["first_emergency"] = row["tick"]
            e["tail"].append(
                dict(
                    tick=row["tick"],
                    speed=speed,
                    source=row["source"],
                    check=row["check"]["reason"],
                    events=row["event_flags"],
                )
            )
            e["tail"] = e["tail"][-10:]
            if row["terminated"] or row["truncated"]:
                e["terminal"] = row["terminal_reason"]
    for counter in [counts, *behavior.values()]:
        counter["stationary_fraction"] = counter["stationary_ticks"] / max(
            counter["ticks"], 1
        )
        counter["bc_stationary_fraction"] = counter["bc_stationary"] / max(
            counter["bc_eligible"], 1
        )
        counter["bc_eligible_fraction"] = counter["bc_eligible"] / max(
            counter["valid_transitions"], 1
        )
    return dict(
        scope="COLLECTION_TRACE_AUDIT",
        versions=metadata.get("versions"),
        stationarity_definition="actual post-step speed <0.1 m/s; BC stationary uses its pre-step measured anchor speed",
        snapshot_bytes=limit,
        complete_lines_sha256=digest.hexdigest(),
        counts=dict(counts),
        by_behavior={k: dict(v) for k, v in behavior.items()},
        unique_scenes=len(scenes),
        scene_ticks=dict(scenes),
        top_scene_fraction=max(scenes.values(), default=0) / max(counts["ticks"], 1),
        outcomes=dict(
            Counter(e["terminal"] or "IN_PROGRESS" for e in episodes.values())
        ),
        episodes=episodes,
        limitation="A stopped-ego collision is an observation, not identification of the colliding object; use exact replay for attribution.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("counts", "by_behavior", "unique_scenes", "outcomes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
