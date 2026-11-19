"""Continuation must preserve model/collection lineage and complete history."""
import json

import pytest

from scripts.sac_diagnostic_continuation import (
    MAX_CONTINUATION_UPDATES, validate_source, continuation_lineage,
    validate_replay_capacity,
)
from src.training.sac.checkpoint import file_hash


@pytest.fixture
def completed(tmp_path):
    arm = dict(mean_gain=.001, mean_basis="integrated", mean_coordinate_scale="entropy",
               actor_head_layer_norm=True, zero_q=True, critic_action_coordinates="whitened",
               critic_action_kernel_init="zero", critic_action_gain=1., critic_factory_sha256="factory")
    collection = dict(mode="actual_failure_branches", samples=8, window=6,
                      selection="moving_rejections", branch_script_sha256="branch", fork_script_sha256="fork")
    for name in ("online-learner.msgpack", "replay.npz"):
        (tmp_path/name).write_bytes(b"test fixture, no learned weights")
    for name in ("ticks.jsonl", "updates.jsonl"):
        (tmp_path/name).write_text('\n'.join(json.dumps(dict(update=i)) for i in (1,2)))
    arm.update(status="COMPLETE", updates=2, learner_sha256=file_hash(tmp_path/"online-learner.msgpack"))
    source = dict(status="COMPLETE", versions={"test":"v1"}, inherited_self_warmup=5000,
                  updates=2, new_transitions=2, collection=collection, arms={"online":arm},
                  saved_replay=dict(size=5002, own_warmup=5000, new_online=2,
                                    sha256=file_hash(tmp_path/"replay.npz")))
    (tmp_path/"report.json").write_text(json.dumps(source))
    return tmp_path,source,arm,collection


def check(fixture, total=3):
    path,source,arm,collection = fixture
    return validate_source(path, source["versions"], arm, collection, total)


def test_lineage_preserves_cumulative_count_and_explicit_collection_fork(completed):
    source = check(completed)
    result = continuation_lineage(completed[0], source, 1002)
    assert result["inherited_online_updates"] == 2
    assert result["collection_seed"] == 1002
    assert set(result["source_hashes"]) == {"report.json", "online-learner.msgpack", "replay.npz", "ticks.jsonl", "updates.jsonl"}


@pytest.mark.parametrize("total", [2, MAX_CONTINUATION_UPDATES + 1])
def test_rejects_nonincreasing_or_excess_budget(completed, total):
    with pytest.raises(ValueError, match="budget"):
        check(completed, total)


def test_larger_budget_preserves_completed_source_and_history(completed):
    path, source, _, _ = completed
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    assert check(completed, total=10_000) == source
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}


@pytest.mark.parametrize("capacity,total", [(50_000, 45_000), (10_000, 5000), (6000, 1001)])
def test_budget_cannot_overwrite_or_wrap_replay_prefix(capacity, total):
    with pytest.raises(ValueError, match="without cursor wrap"):
        validate_replay_capacity(capacity, total)


@pytest.mark.parametrize("capacity,total", [(50_000, 10_000), (10_001, 5000), (6001, 1000)])
def test_budget_with_preserved_prefix_capacity(capacity, total):
    validate_replay_capacity(capacity, total)


def test_rejects_changed_model_or_collector(completed):
    completed[2]["critic_action_gain"] = .1
    with pytest.raises(ValueError, match="model setting"):
        check(completed)
    completed[2]["critic_action_gain"] = 1.
    completed[3]["selection"] = "recent"
    with pytest.raises(ValueError, match="collection setting"):
        check(completed)


def test_rejects_corrupted_artifact_or_missing_history(completed):
    path = completed[0]
    p = path/"replay.npz"
    old = p.read_bytes();p.write_bytes(old+b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        check(completed)
    p.write_bytes(old)
    (path/"ticks.jsonl").write_text('{"update":2}\n')
    with pytest.raises(ValueError, match="history"):
        check(completed)


def test_rollout_horizon_and_implementation_cannot_change_on_resume(completed):
    completed[3]["rollout_horizon"] = 1
    completed[3]["rollout_script_sha256"] = None
    check(completed)  # Historical files omitted both optional fields.
    completed[3]["rollout_horizon"] = 8
    with pytest.raises(ValueError, match="rollout_horizon"):
        check(completed)
    completed[3]["rollout_horizon"] = 1
    completed[3]["rollout_script_sha256"] = "different implementation"
    with pytest.raises(ValueError, match="rollout_script_sha256"):
        check(completed)
    completed[3]["rollout_script_sha256"] = None
    completed[3]["restart_terminal"] = True
    with pytest.raises(ValueError, match="restart_terminal"):
        check(completed)
    completed[3]["restart_terminal"] = False
    completed[3]["complete_path"] = True
    with pytest.raises(ValueError, match="complete_path"):
        check(completed)
    completed[3]["complete_path"] = False
    completed[3]["extend_selected"] = True
    with pytest.raises(ValueError, match="extend_selected"):
        check(completed)
    completed[3]["extend_selected"] = False
    completed[3]["quota_script_sha256"] = "different quota implementation"
    with pytest.raises(ValueError, match="quota_script_sha256"):
        check(completed)


@pytest.fixture
def budget_resume(completed):
    import copy
    path, source, _, _ = completed
    current = path / "continued"
    current.mkdir()
    rows = [dict(update=i + 1, episode_uid="old" if i < 2 else "new",
                 pre_tick=i if i < 2 else i - 2,
                 lineage={"kind": "failure_branch" if i == 1 else "main"})
            for i in range(4)]
    for name, values in (("ticks.jsonl", rows), ("updates.jsonl", [dict(update=i+1) for i in range(4)])):
        lines = [json.dumps(row) + "\n" for row in values]
        (path / name).write_text("".join(lines[:2]))
        (current / name).write_text("".join(lines))
    report = copy.deepcopy(source)
    report.update(updates=4, new_transitions=4, inherited_online_updates=2,
                  continuation_history=[continuation_lineage(path, source, 2)])
    report["arms"]["online"]["updates"] = 4
    return current, report, rows


def test_budget_audit_accepts_only_verified_inherited_and_final_boundaries(budget_resume):
    from scripts.audit_sac_branch_replay import validated_budget_end_indices
    current, report, rows = budget_resume
    assert validated_budget_end_indices(current, report, rows) == {1, 3}
    report.pop("continuation_history")
    assert validated_budget_end_indices(current, report, rows) == {3}


@pytest.mark.parametrize("damage", ["hash", "prefix", "count", "restart", "missing_hash", "duplicate"])
def test_budget_audit_rejects_unverified_cutoffs(budget_resume, damage):
    from scripts.audit_sac_branch_replay import validated_budget_end_indices
    current, report, rows = budget_resume
    entry = report["continuation_history"][0]
    if damage == "hash":
        entry["source_hashes"]["ticks.jsonl"] = "invalid"
    elif damage == "prefix":
        (current / "ticks.jsonl").write_text((current / "ticks.jsonl").read_text().replace('"old"', '"changed"', 1))
    elif damage == "count":
        entry["inherited_online_updates"] = 1
    elif damage == "restart":
        rows[2]["pre_tick"] = 2
    elif damage == "missing_hash":
        entry["source_hashes"].pop("replay.npz")
    else:
        report["continuation_history"].append(entry.copy())
    with pytest.raises(ValueError):
        validated_budget_end_indices(current, report, rows)
