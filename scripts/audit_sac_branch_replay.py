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


def validated_budget_end_indices(directory, report, rows):
    """Allow only final or hash-verified inherited budget cutoffs, never arbitrary gaps."""
    from scripts.sac_diagnostic_continuation import validate_source
    from src.training.sac.checkpoint import file_hash

    ends = {len(rows) - 1}
    history = report.get("continuation_history", [])
    expected_files = {"report.json", "online-learner.msgpack", "replay.npz",
                      "ticks.jsonl", "updates.jsonl"}
    previous_count = 0
    for position, entry in enumerate(history):
        source_dir = Path(entry["source_directory"])
        hashes = entry["source_hashes"]
        if set(hashes) != expected_files or any(
                file_hash(source_dir / name) != digest for name, digest in hashes.items()):
            raise ValueError("inherited budget source checksum mismatch")
        source = validate_source(source_dir, report["versions"], report["arms"]["online"],
                                 report["collection"], report["updates"])
        count = entry["inherited_online_updates"]
        if (type(count) is not int or not previous_count < count < len(rows)
                or count != source["updates"]
                or source.get("continuation_history", []) != history[:position]):
            raise ValueError("invalid inherited budget history")
        for name in ("ticks.jsonl", "updates.jsonl"):
            prefix = (source_dir / name).read_bytes().splitlines(keepends=True)
            current = (directory / name).read_bytes().splitlines(keepends=True)
            if len(prefix) != count or current[:count] != prefix:
                raise ValueError("inherited budget history prefix mismatch")
        following = rows[count]
        if (following["lineage"]["kind"] != "main" or following["pre_tick"] != 0
                or following["episode_uid"] == rows[count - 1]["episode_uid"]):
            raise ValueError("inherited budget must explicitly restart collection environment")
        ends.add(count - 1)
        previous_count = count
    if history and previous_count != report.get("inherited_online_updates"):
        raise ValueError("inherited budget count mismatch")
    return ends


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
    parents, branch_pairs, successor_pairs, branches = {}, [], [], []
    branch_ids, states_per_trigger = set(), {}
    horizon = report["collection"].get("rollout_horizon", 1)
    restart = report["collection"].get("restart_terminal", False)
    complete_path = report["collection"].get("complete_path", False)
    extend_selected = report["collection"].get("extend_selected", False)
    assert all(isinstance(flag, bool) for flag in (restart, complete_path, extend_selected))
    if extend_selected:
        assert restart and not complete_path and horizon == 8
        assert report["collection"]["samples"] == 8 and report["collection"]["window"] == 6
        assert report["collection"]["selection"] == "moving_predecessors"
        assert report["collection"].get("quota_script_sha256")
    assert isinstance(horizon, int) and 1 <= horizon <= (200 if complete_path else 8)
    assert not restart or horizon > 1
    if complete_path:
        assert not restart and horizon == 200
        assert report["collection"]["samples"] == 1 and report["collection"]["window"] == 6
        assert report["collection"]["selection"] == "moving_predecessors"
        assert report["collection"].get("rollout_script_sha256")
    elif horizon > 1:
        assert report["collection"]["samples"] == (horizon if restart else 1)
        assert report["collection"]["window"] * horizon <= 48
        assert report["collection"].get("rollout_script_sha256")
    root_counts, root_latest, root_initial_lineage, extended_roots = {}, {}, {}, {}
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
            root_key = (lineage["parent_episode_uid"], lineage["parent_tick"])
            assert row["episode_uid"] == root_key[0] and root_key in parents
            depth = lineage.get("branch_step", 0)
            assert isinstance(depth, int) and 0 <= depth < (200 if extend_selected else horizon)
            assert row["pre_tick"] == root_key[1] + depth
            branch_id = (*root_key, lineage["branch_sample"], depth)
            assert branch_id not in branch_ids
            branch_ids.add(branch_id)
            assert 0 <= lineage["branch_sample"] < report["collection"]["samples"]
            assert row["scene_id"] == rows[parents[root_key]]["scene_id"]
            assert row["provider"] == rows[parents[root_key]]["provider"]
            if restart:
                quota_key = (*root_key, lineage["trigger_tick"])
                used = root_counts.get(quota_key, 0)
                assert lineage["root_budget_step"] == used
                assert used < horizon or (extend_selected and used < horizon + 200)
                is_extension = used >= horizon
                assert lineage.get("quota_extension", False) == is_extension
                if is_extension:
                    trigger_key = row["episode_uid"], lineage["trigger_tick"]
                    assert lineage["root_selection_rule"] == "earliest_predecessor_then_rejection_then_recent"
                    assert all(root_counts.get((*key, lineage["trigger_tick"]), 0) >= horizon
                               for key in states_per_trigger[trigger_key])
                    assert extended_roots.setdefault(trigger_key, root_key) == root_key
                if used == 0:
                    assert depth == 0 and lineage["branch_sample"] == 0
                    root_initial_lineage[root_key] = lineage
                else:
                    previous_root = rows[root_latest[quota_key]]
                    if is_extension:
                        assert previous_root["bootstrap"]
                    if previous_root["bootstrap"]:
                        assert lineage["previous_collection_id"] == previous_root["collection_id"]
                    else:
                        assert depth == 0 and lineage["previous_collection_id"] is None
                        assert lineage["branch_sample"] == previous_root["lineage"]["branch_sample"] + 1
                root_counts[quota_key] = used + 1
                root_latest[quota_key] = i
            if depth == 0:
                assert lineage.get("previous_collection_id") is None
                branch_pairs.append((i + 5000, parents[root_key] + 5000))
                if horizon > 1:
                    assert lineage["root_observation_sha256"] == lineage["observation_sha256"]
            else:
                previous_id = lineage["previous_collection_id"]
                assert isinstance(previous_id, int) and 1 <= previous_id <= i
                previous = rows[previous_id - 1]
                pl = previous["lineage"]
                for field in ("kind", "parent_episode_uid", "parent_tick", "branch_sample", "trigger_tick",
                              "trigger_outcome", "state_selection", "selection_role", "root_observation_sha256"):
                    assert pl[field] == lineage[field]
                assert pl.get("predecessor_rejection_tick") == lineage.get("predecessor_rejection_tick")
                assert pl["branch_step"] + 1 == depth
                assert previous["bootstrap"] and not previous["terminated"] and not previous["truncated"]
                assert previous["pre_tick"] + 1 == row["pre_tick"]
                assert previous["episode_uid"] == row["episode_uid"]
                successor_pairs.append((i + 5000, previous_id - 1 + 5000))
            assert lineage["trigger_outcome"] in ("COLLISION", "OFFROAD")
            trigger_key = row["episode_uid"], lineage["trigger_tick"]
            states_per_trigger.setdefault(trigger_key, set()).add(root_key)
            assert len(states_per_trigger[trigger_key]) <= report["collection"]["window"]
            if complete_path:
                assert len(states_per_trigger[trigger_key]) == 1
                assert lineage["root_selection_rule"] == "earliest_predecessor_then_rejection_then_recent"
            assert trigger_key in parents
            trigger = rows[parents[trigger_key]]
            assert not trigger["bootstrap"] and trigger["terminal_reason"] == lineage["trigger_outcome"]
            assert 0 <= lineage["trigger_tick"] - root_key[1]
            if report["collection"].get("selection", "recent") == "recent":
                assert lineage["trigger_tick"] - root_key[1] < report["collection"]["window"]
            else:
                selection = report["collection"]["selection"]
                assert selection in ("moving_rejections", "moving_predecessors")
                assert lineage["state_selection"] == selection
                assert lineage["selection_role"] in ("moving_rejection", "moving_predecessor", "terminal", "recent")
                if lineage["selection_role"] == "terminal":
                    assert root_key[1] == lineage["trigger_tick"]
                if lineage["selection_role"] == "moving_rejection":
                    assert not rows[parents[root_key]]["check_valid"]
                if lineage["selection_role"] == "moving_predecessor":
                    assert selection == "moving_predecessors"
                    assert rows[parents[root_key]]["check_valid"]
                    rejected_tick = lineage["predecessor_rejection_tick"]
                    assert 0 < rejected_tick - root_key[1] < report["collection"]["window"]
                    rejected_key = row["episode_uid"], rejected_tick
                    assert rejected_key in parents
                    assert not rows[parents[rejected_key]]["check_valid"]
            branches.append(row)
        assert row["delta_ticks"] == 1 and row["tick"] == row["pre_tick"] + 1
        assert row["bootstrap"] == (not (row["terminated"] or row["truncated"]))
    priority = {"moving_predecessor": 0, "moving_rejection": 1, "recent": 2, "terminal": 3}
    budget_ends = validated_budget_end_indices(directory, report, rows) if extend_selected else {len(rows) - 1}
    for trigger, selected_root in extended_roots.items():
        expected = min(states_per_trigger[trigger], key=lambda key: (
            priority[root_initial_lineage[key]["selection_role"]], key[1]))
        assert selected_root == expected
        last_index = root_latest[(*selected_root, trigger[1])]
        assert not rows[last_index]["bootstrap"] or last_index in budget_ends
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
            next_array = data["next_" + key]
            for branch, previous in successor_pairs:
                np.testing.assert_array_equal(array[branch], next_array[previous])
        # ego.speed is field2 after the three task fields, normalized by30.
        if report["collection"].get("selection") in ("moving_rejections", "moving_predecessors"):
            obs = data["obs"]
            for i, row in enumerate(rows):
                if row["lineage"].get("selection_role") in ("moving_rejection", "moving_predecessor"):
                    root = (row["lineage"]["parent_episode_uid"], row["lineage"]["parent_tick"])
                    assert obs[5000+parents[root], 5] * 30 >= 0.1 - 1e-6
                if row["lineage"].get("selection_role") == "moving_predecessor":
                    rejected_key = row["episode_uid"], row["lineage"]["predecessor_rejection_tick"]
                    assert obs[5000+parents[rejected_key], 5] * 30 >= 0.1 - 1e-6
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
                branch_unique_parent_states=len({(r["lineage"]["parent_episode_uid"],r["lineage"]["parent_tick"]) for r in branches}),
                branch_continuation_transitions=len(successor_pairs), rollout_horizon=horizon,
                restart_terminal=restart, complete_path=complete_path,
                extend_selected=extend_selected,
                extended_roots=len(extended_roots),
                validated_budget_end_collection_ids=sorted(i + 1 for i in budget_ends),
                extension_transitions=sum(r["lineage"].get("quota_extension", False) for r in branches),
                root_step_counts=list(root_counts.values()) if restart else None,
                main_provider_counts=dict(Counter(r["provider"] for r in rows if r["lineage"]["kind"]=="main")),
                branch_provider_counts=dict(Counter(r["provider"] for r in branches)),
                branch_rejections=sum(not r["check_valid"] for r in branches),
                branch_outcomes=dict(Counter(r["terminal_reason"] or "NONTERMINAL" for r in branches)),
                checks="One actual logged step per update; exact replay action/reward/termination/lineage; branch root obs/masks equal main parent; continuation obs/masks equal previous actual next state; teacher0; all replay scenes and new scene/provider pairs in hash-verified train manifest.")


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
