"""Continuation must preserve model/collection lineage and complete history."""
import json

import pytest

from scripts.sac_diagnostic_continuation import validate_source, continuation_lineage
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


@pytest.mark.parametrize("total", [2,5001])
def test_rejects_nonincreasing_or_excess_budget(completed, total):
    with pytest.raises(ValueError, match="budget"):
        check(completed, total)


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
