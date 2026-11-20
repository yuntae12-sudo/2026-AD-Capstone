"""Audit cumulative actual refresh lineage after a verified context continuation."""
import json
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np


def audit(directory, manifest):
    from scripts.sac_branch_reconstruction import branch_history_indices
    from src.training.sac.checkpoint import file_hash

    read = lambda p: json.loads(p.read_text())
    report = read(directory / "report.json")
    from scripts.sac_refresh_context_provenance import validate_provenance
    provenance = validate_provenance(directory, manifest)
    early_moving = provenance["contract"] == "early_moving_refresh"
    if early_moving:
        from scripts.sac_early_moving_refresh import refresh_spec
    else:
        from scripts.sac_provider_protected_refresh import refresh_spec
    assert report[provenance["contract"]] == refresh_spec()
    assert file_hash(directory / "replay.npz") == report["saved_replay"]["sha256"]
    assert file_hash(manifest) == report["versions"]["manifest_sha256"]
    allowed = {(r["scene_id"], r["provider"]) for r in read(manifest)["rows"]
               if r["split"] == "train" and r["status"] == "RESET_ACCEPTED"}
    rows = [json.loads(x) for x in (directory / "ticks.jsonl").read_text().splitlines()]
    logs = [json.loads(x) for x in (directory / "updates.jsonl").read_text().splitlines()]
    assert len(rows) == len(logs) == report["updates"] == report["new_transitions"]
    with np.load(directory / "replay.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files if k != "metadata"}
        meta = json.loads(str(archive["metadata"]))
    assert meta["size"] == meta["cursor"] == 5000 + len(rows)
    assert meta["versions"] == report["versions"] and not arrays["teacher_accepted"].any()
    assert set(arrays["scene_id"]) <= {s for s, _ in allowed}
    for key, field in [("raw_u", None), ("reward", "reward"), ("bootstrap", "bootstrap"),
                       ("terminated", "terminated"), ("truncated", "truncated"),
                       ("tick", "pre_tick"), ("episode_uid", "episode_uid"), ("scene_id", "scene_id")]:
        values = [r[field] if field else r["raw_proposal"]["raw_u"] for r in rows]
        np.testing.assert_array_equal(arrays[key][5000:], values)

    def same_observation(i, j, previous_next=False):
        for key in ("obs", "fields", "items"):
            np.testing.assert_array_equal(arrays[key][5000+i],
                arrays[("next_" if previous_next else "")+key][5000+j])

    banks = {"terminal": [], "nonterminal": []}
    cursors = dict.fromkeys(banks, 0)
    next_stage = "terminal"
    next_provider = "fixed_keep"
    anchors = {}
    active = None
    bases, refreshes, expected_refresh = 0, 0, False
    parents, main_last, path_last = {}, {}, {}
    counts = Counter()
    config = report["collection"]
    assert config["complete_path"] and config["samples"] == 1 and config["window"] == 6
    assert config["selection"] == "moving_predecessors" and config["rollout_horizon"] == 200
    assert not config["restart_terminal"] and not config["extend_selected"]
    assert config["rollout_script_sha256"] == file_hash(Path("scripts/sac_failure_complete_path.py"))
    assert config["branch_script_sha256"] == file_hash(Path("scripts/sac_failure_branches.py"))
    assert config["fork_script_sha256"] == file_hash(Path("scripts/diagnose_sac_q_values.py"))
    for i, (row, log) in enumerate(zip(rows, logs)):
        assert row["collection_id"] == row["update"] == log["update"] == i+1
        assert row["policy_updates_before"] == i
        assert (row["scene_id"], row["provider"]) in allowed
        assert row["tick"] == row["pre_tick"]+1 and row["delta_ticks"] == 1
        assert row["bootstrap"] == (not (row["terminated"] or row["truncated"]))
        lineage = row["lineage"]; kind = lineage["kind"]; counts[kind] += 1
        assert (kind == "successor_refresh") == expected_refresh
        if kind == "successor_refresh":
            other = "nonterminal" if next_stage == "terminal" else "terminal"
            available = lambda name: bool(banks[name]) or (name == "nonterminal" and active is not None)
            stage = next_stage if available(next_stage) else other
            bank = banks[stage]
            continuing = stage == "nonterminal" and active is not None
            if continuing:
                source_id, path_id, path_step, previous_id = active
            else:
                if early_moving and stage == "nonterminal":
                    available_commands = {rows[j-1]["provider"] for j in bank}
                    other_provider = "fixed_follow" if next_provider == "fixed_keep" else "fixed_keep"
                    provider = next_provider if next_provider in available_commands else other_provider
                    source_id = anchors.get(provider)
                    if source_id is None:
                        source_id = max(j for j in bank if rows[j-1]["provider"] == provider)
                    next_provider = "fixed_follow" if provider == "fixed_keep" else "fixed_keep"
                else:
                    source_id = bank[cursors[stage] % len(bank)]
                    cursors[stage] = (cursors[stage]+1) % len(bank)
                path_id, path_step, previous_id = refreshes+1, 0, None
            assert lineage["refresh_path_id"] == path_id
            assert lineage["refresh_path_step"] == path_step
            assert lineage["previous_refresh_collection_id"] == previous_id
            next_stage = "nonterminal" if stage == "terminal" else "terminal"
            refreshes += 1; expected_refresh = False
            assert lineage["source_stage"] == stage
            assert lineage["bank_collection_ids"] == bank
            assert lineage["stage_bank_collection_ids"] == banks
            assert lineage["source_collection_id"] == source_id
            assert lineage["base_steps"] == bases and lineage["refresh_index"] == refreshes
            source = rows[source_id-1]
            if early_moving and stage == "nonterminal":
                speed = math.hypot(*source["raw_proposal"]["anchor_velocity_world"])
                assert lineage["source_speed_mps"] == speed
                assert lineage["root_selection"] == ("early_moving" if speed >= .1 else "latest_fallback")
            assert source["lineage"]["kind"] in ("main", "failure_branch")
            assert ((source["bootstrap"] and not source["check_valid"])
                    or (not source["bootstrap"] and source["terminal_reason"] in ("COLLISION", "OFFROAD")))
            assert lineage["source_outcome"] == source["terminal_reason"]
            assert lineage["source_check_valid"] == source["check_valid"]
            assert lineage["source_check_reason"] == source["check_reason"]
            for field in ("scene_id", "provider", "episode_uid"):
                assert row[field] == source[field]
            assert row["pre_tick"] == source["pre_tick"] + path_step
            assert lineage["scene_id"] == row["scene_id"] and lineage["provider"] == row["provider"]
            if continuing:
                assert rows[previous_id-1]["bootstrap"]
                same_observation(i, previous_id-1, True)
            else:
                same_observation(i, source_id-1)
            actual_fingerprint = hashlib.sha256()
            for field in ("obs", "fields", "items"):
                value = arrays[field][5000+i]
                actual_fingerprint.update(str((value.dtype.str, value.shape)).encode())
                actual_fingerprint.update(value.tobytes())
            assert lineage["refresh_pre_observation_sha256"] == actual_fingerprint.hexdigest()
            fingerprint = hashlib.sha256()
            for field in ("obs", "fields", "items"):
                value = arrays[field][5000+source_id-1]
                fingerprint.update(str((value.dtype.str, value.shape)).encode())
                fingerprint.update(value.tobytes())
            assert lineage["source_observation_sha256"] == fingerprint.hexdigest()
            if stage == "nonterminal":
                active = (source_id, path_id, path_step+1, i+1) if row["bootstrap"] else None
            continue
        assert kind in ("main", "failure_branch")
        if kind == "main":
            key = row["episode_uid"], row["pre_tick"]
            assert key not in parents; parents[key] = i
            previous = main_last.get(row["episode_uid"])
            if previous is None:
                assert row["pre_tick"] == 0
            else:
                assert rows[previous]["bootstrap"] and rows[previous]["pre_tick"]+1 == row["pre_tick"]
                same_observation(i, previous, True)
            main_last[row["episode_uid"]] = i
        else:
            history = branch_history_indices(rows, i)
            assert [rows[j]["pre_tick"] for j in history] == list(range(row["pre_tick"]+1))
            root = lineage["parent_episode_uid"], lineage["parent_tick"]
            trigger = lineage["parent_episode_uid"], lineage["trigger_tick"]
            assert trigger in parents and parents[trigger] < i
            assert not rows[parents[trigger]]["bootstrap"]
            assert lineage["trigger_outcome"] == rows[parents[trigger]]["terminal_reason"]
            assert lineage["trigger_outcome"] in ("COLLISION", "OFFROAD")
            assert lineage["branch_sample"] == 0 and 0 <= lineage["branch_step"] < 200
            assert lineage["root_selection_rule"] == "earliest_predecessor_then_rejection_then_recent"
            if lineage["branch_step"] == 0:
                same_observation(i, parents[root])
                assert lineage["state_selection"] == "moving_predecessors"
                role = lineage["selection_role"]
                assert role in ("moving_predecessor", "moving_rejection", "recent", "terminal")
                parent = rows[parents[root]]
                if role in ("moving_predecessor", "moving_rejection"):
                    assert arrays["obs"][5000+parents[root], 5]*30 >= .1-1e-6
                if role == "moving_predecessor":
                    assert parent["check_valid"]
                    rejected = (root[0], lineage["predecessor_rejection_tick"])
                    assert rejected in parents and not rows[parents[rejected]]["check_valid"]
                    assert 0 < rejected[1]-root[1] < 6
                elif role == "moving_rejection":
                    assert not parent["check_valid"]
                elif role == "terminal":
                    assert root == trigger
            else:
                same_observation(i, lineage["previous_collection_id"]-1, True)
            previous_path = path_last.get(trigger)
            if previous_path is not None:
                assert lineage["previous_collection_id"] == previous_path+1
            else:
                assert lineage["branch_step"] == 0
            path_last[trigger] = i
        bases += 1
        if ((row["bootstrap"] and not row["check_valid"])
                or (not row["bootstrap"] and row["terminal_reason"] in ("COLLISION", "OFFROAD"))):
            stage = "nonterminal" if row["bootstrap"] else "terminal"
            bank = banks[stage]
            bank.append(i+1)
            if early_moving and stage == "nonterminal":
                speed = math.hypot(*row["raw_proposal"]["anchor_velocity_world"])
                assert math.isfinite(speed)
                np.testing.assert_allclose(arrays["obs"][5000+i, 5],
                                           np.clip(speed/30, -5, 5), rtol=0, atol=1e-6)
                provider = row["provider"]
                assert provider in ("fixed_keep", "fixed_follow")
                previous_anchor = rows[anchors[provider]-1] if provider in anchors else None
                if speed >= .1 and (previous_anchor is None or
                        previous_anchor["episode_uid"] != row["episode_uid"] or
                        row["pre_tick"] <= previous_anchor["pre_tick"]):
                    anchors[provider] = i+1
            if len(bank) > 4:
                if stage == "nonterminal":
                    providers = [rows[j-1]["provider"] for j in bank]
                    assert set(providers) <= {"fixed_keep", "fixed_follow"}
                    overrepresented = {p for p in providers if providers.count(p) > 2}
                    victim = min(j for j in bank if rows[j-1]["provider"] in overrepresented
                                 and j not in anchors.values())
                    position = bank.index(victim)
                else:
                    position = 0
                bank.pop(position)
                if position < cursors[stage]:
                    cursors[stage] -= 1
            assert set(anchors.values()) <= set(banks["nonterminal"])
        expected_refresh = (any(banks.values()) or active is not None) and bases % 4 == 0
    for i in path_last.values():
        assert not rows[i]["bootstrap"] or all(r["lineage"]["kind"] == "successor_refresh" for r in rows[i+1:])
    assert counts["main"] == config["main_transitions"]
    assert counts["failure_branch"] == config["branch_transitions"]
    assert counts["successor_refresh"] == report["refresh_transitions"] == refreshes
    assert refreshes <= bases // 4
    assert report["refresh_terminal_transitions"] == sum(
        not r["bootstrap"] for r in rows if r["lineage"]["kind"] == "successor_refresh")
    assert config["branch_terminal_transitions"] == sum(
        not r["bootstrap"] for r in rows if r["lineage"]["kind"] == "failure_branch")
    episode_groups = {}
    for row in rows:
        if row["lineage"]["kind"] == "main":
            episode_groups.setdefault(row["episode_uid"], []).append(row)
    assert len(episode_groups) == len(report["episodes"])
    for recorded, actual in zip(report["episodes"], episode_groups.values()):
        assert recorded["scene_id"] == actual[0]["scene_id"] and recorded["provider"] == actual[0]["provider"]
        assert recorded["ticks"] == len(actual) and recorded["accepted"] == sum(r["check_valid"] for r in actual)
        assert recorded["outcome"] == (actual[-1]["terminal_reason"] or "DIAGNOSTIC_BUDGET")
        np.testing.assert_allclose(recorded["return_training"], sum(r["reward"] for r in actual), rtol=0, atol=1e-10)
    return dict(status="COMPLETE", qualified=False, provenance=provenance, counts=dict(counts), base_steps=bases,
        refresh_steps=refreshes, exact_refresh_parent_observations=refreshes, active_path_at_budget=active,
        actual_successor_steps=sum(r["lineage"].get("refresh_path_step", 0) > 0 for r in rows),
        unique_refresh_sources=len({r["lineage"]["source_collection_id"] for r in rows
                                    if r["lineage"]["kind"] == "successor_refresh"}),
        refresh_outcomes=dict(Counter(r["terminal_reason"] or "NONTERMINAL" for r in rows
                                     if r["lineage"]["kind"] == "successor_refresh")),
        note="Exact train lineage and flags; independent declared anchor/command/root reconstruction across the original budget cut; terminal FIFO, actual successor arrays and waiting path lineage; no invented terminal.")
