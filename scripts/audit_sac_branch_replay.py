#!/usr/bin/env python3
"""Check saved real branch lineage, update accounting and train-only replay."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(directory, manifest):
    from src.training.sac.checkpoint import file_hash

    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "COMPLETE":
        raise ValueError("requires completed collection")
    if file_hash(directory / "replay.npz") != report["saved_replay"]["sha256"]:
        raise ValueError("replay checksum mismatch")
    rows = [json.loads(line) for line in (directory / "ticks.jsonl").read_text().splitlines()]
    if file_hash(manifest) != report["versions"]["manifest_sha256"]:
        raise ValueError("manifest checksum mismatch")
    inventory = json.loads(manifest.read_text())["rows"]
    allowed = {(r["scene_id"], r["provider"]) for r in inventory
               if r["split"] == "train" and r["status"] == "RESET_ACCEPTED"}
    assert all((r["scene_id"], r["provider"]) in allowed for r in rows)
    updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    count = report["new_transitions"]
    assert count == len(rows) == len(updates) == report["updates"]
    parents, branch_pairs, branches = {}, [], []
    for i, (row, update) in enumerate(zip(rows, updates)):
        assert row["collection_id"] == row["update"] == update["update"] == i + 1
        assert row["policy_updates_before"] == i
        key = row["episode_uid"], row["pre_tick"]
        if row["lineage"]["kind"] == "main":
            assert key not in parents
            parents[key] = i
        else:
            lineage = row["lineage"]
            assert lineage["kind"] == "failure_branch"
            assert key == (lineage["parent_episode_uid"], lineage["parent_tick"])
            assert key in parents
            assert row["scene_id"] == rows[parents[key]]["scene_id"]
            assert row["provider"] == rows[parents[key]]["provider"]
            assert lineage["trigger_outcome"] in ("COLLISION", "OFFROAD")
            trigger_key = row["episode_uid"], lineage["trigger_tick"]
            assert trigger_key in parents
            trigger = rows[parents[trigger_key]]
            assert not trigger["bootstrap"] and trigger["terminal_reason"] == lineage["trigger_outcome"]
            assert 0 <= lineage["trigger_tick"] - row["pre_tick"] < report["collection"]["window"]
            branch_pairs.append((i + 5000, parents[key] + 5000))
            branches.append(row)
        assert row["delta_ticks"] == 1 and row["tick"] == row["pre_tick"] + 1
        assert row["bootstrap"] == (not (row["terminated"] or row["truncated"]))
    with np.load(directory / "replay.npz", allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        assert meta["versions"] == report["versions"]
        assert meta["size"] == meta["cursor"] == 5000 + count
        assert not data["teacher_accepted"].any()
        assert set(data["scene_id"].tolist()) <= {s for s, _ in allowed}
        for key, values in (
            ("raw_u", [r["raw_proposal"]["raw_u"] for r in rows]),
            ("reward", [r["reward"] for r in rows]),
            ("bootstrap", [r["bootstrap"] for r in rows]),
            ("terminated", [r["terminated"] for r in rows]),
            ("truncated", [r["truncated"] for r in rows]),
            ("tick", [r["pre_tick"] for r in rows]),
            ("episode_uid", [r["episode_uid"] for r in rows]),
            ("scene_id", [r["scene_id"] for r in rows]),
        ):
            np.testing.assert_array_equal(data[key][5000:], values)
        for key in ("obs", "fields", "items"):
            array = data[key]
            for branch, parent in branch_pairs:
                np.testing.assert_array_equal(array[branch], array[parent])
    main_count = len(rows) - len(branches)
    terminal_count = sum(not r["bootstrap"] for r in branches)
    assert main_count == report["collection"]["main_transitions"]
    assert len(branches) == report["collection"]["branch_transitions"]
    assert terminal_count == report["collection"]["branch_terminal_transitions"]
    return dict(status="COMPLETE", qualified=False, report_sha256=file_hash(directory / "report.json"),
                replay_sha256=report["saved_replay"]["sha256"],
                new_transitions=count, main_transitions=main_count, branch_transitions=len(branches),
                branch_terminal_transitions=terminal_count, branch_bootstrap_transitions=len(branches)-terminal_count,
                unique_train_scenes=len({r["scene_id"] for r in rows}),
                branch_unique_parent_states=len({(r["episode_uid"],r["pre_tick"]) for r in branches}),
                main_provider_counts=dict(Counter(r["provider"] for r in rows if r["lineage"]["kind"]=="main")),
                branch_provider_counts=dict(Counter(r["provider"] for r in branches)),
                branch_rejections=sum(not r["check_valid"] for r in branches),
                branch_outcomes=dict(Counter(r["terminal_reason"] or "NONTERMINAL" for r in branches)),
                checks="One actual logged step per update; exact replay action/reward/termination/lineage; branch obs/masks equal actual parent; teacher0; all replay scenes and new scene/provider pairs in hash-verified train manifest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.directory, args.manifest)
    result["audit_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
