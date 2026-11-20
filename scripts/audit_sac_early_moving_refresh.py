#!/usr/bin/env python3
"""Audit fresh rejected-state refresh with unchanged terminal sampler/full updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(directory, checkpoint, manifest):
    import numpy as np
    from flax import serialization
    from scripts.audit_sac_early_moving_replay import audit as branch_audit
    from scripts.sac_terminal_retention_sampling import sampling_spec, draw_indices, terminal_pools, FAILURE_REASONS
    from scripts.sac_diagnostic_sampling import indices_hash
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = read(directory / "report.json")
    assert report["status"] == "COMPLETE"
    assert report["scope"] == "OWN_EARLY_MOVING_REFRESH_SAC_COMPARISON"
    assert report["replay_sampling"] == sampling_spec()
    assert report["script_sha256"] == sha(Path(__file__).with_name("diagnose_sac_early_moving_refresh.py"))
    assert not report.get("continuation_history") and report["inherited_online_updates"] == 0
    old = branch_audit(directory, manifest)
    checkpoint_meta = read(checkpoint / "manifest.json")
    assert report["source_checkpoint_sha256"] == sha(checkpoint / "manifest.json")
    assert checkpoint_meta["hashes"]["replay.npz"] == sha(checkpoint / "replay.npz")
    goal_ends = []; failure_ends = []; offset = 0
    for episode in checkpoint_meta["run_state"]["report"]["episodes"]:
        offset += episode["valid_transitions"]
        if offset > 5000:
            break
        if episode["outcome"] == "ROUTE_GOAL":
            goal_ends.append(offset - 1)
        if episode["outcome"] in FAILURE_REASONS:
            failure_ends.append(offset - 1)
    with np.load(directory / "replay.npz", allow_pickle=False) as current, np.load(checkpoint / "replay.npz", allow_pickle=False) as initial:
        names = set(current.files) - {"metadata"}
        assert len(names) == 17 and names == set(initial.files) - {"metadata"}
        for name in names:
            assert current[name][:5000].tobytes() == initial[name][:5000].tobytes(), name
        bootstrap, reward = current["bootstrap"].copy(), current["reward"].copy()
        meta = json.loads(str(current["metadata"]))
    warm_goals, warm_failures = terminal_pools(5000, bootstrap, reward)
    np.testing.assert_array_equal(warm_goals, goal_ends)
    np.testing.assert_array_equal(warm_failures, failure_ends)
    ticks = [json.loads(line) for line in (directory / "ticks.jsonl").read_text().splitlines()]
    expected_goal = goal_ends + [5000 + i for i, row in enumerate(ticks) if row["terminal_reason"] == "ROUTE_GOAL"]
    expected_failure = failure_ends + [5000 + i for i, row in enumerate(ticks) if row["terminal_reason"] in FAILURE_REASONS]
    final_goals, final_failures = terminal_pools(meta["size"], bootstrap, reward)
    np.testing.assert_array_equal(final_goals, expected_goal)
    np.testing.assert_array_equal(final_failures, expected_failure)
    logs = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    count = report["updates"]
    rng = np.random.default_rng(report["seed"])
    exposure = np.zeros(meta["size"], np.int64)
    goal_slot_count = 0; failure_slot_count = 0; pool_sizes = []; failure_pool_sizes = []
    assert len(logs) == len(ticks) == count and meta["size"] == 5000 + count
    for step, row in enumerate(logs, 1):
        size = 5000 + step
        indices = draw_indices(rng, size, bootstrap, reward)
        assert row["update"] == step and row["replay_size"] == size
        assert row["replay_indices_sha256"] == indices_hash(indices)
        pool, failure_pool = terminal_pools(size, bootstrap, reward)
        assert ((indices[128:240] >= max(0, size - 256)) & (indices[128:240] < size)).all()
        if len(failure_pool):
            assert np.isin(indices[248:256], failure_pool).all()
            failure_slot_count += 8
        failure_pool_sizes.append(len(failure_pool))
        if len(pool):
            assert np.isin(indices[240:248], pool).all()
            goal_slot_count += 8
        pool_sizes.append(len(pool))
        np.add.at(exposure, indices, 1)
    assert rng.bit_generator.state == meta["rng"] and int(exposure.sum()) == count * 256
    weights = directory / "online-learner.msgpack"
    assert sha(weights) == report["arms"]["online"]["learner_sha256"]
    state = serialization.msgpack_restore(weights.read_bytes())
    assert int(state["updates"]) == count and int(state["bc_updates"]) == 0
    for name in ("actor", "q1", "q2"):
        assert int(state[name]["step"]) == count
    initial = read(directory / "initial-state-audit.json")
    assert initial["status"] == "COMPLETE" and initial["updates"] == initial["bc_updates"] == 0
    assert sha(directory / "initial-learner.msgpack") == initial["learner_sha256"]
    if initial["reference_file"] is not None:
        assert initial["reference_bytes_equal"] is True
        assert sha(Path(initial["reference_file"])) == initial["reference_sha256"] == initial["learner_sha256"]
    return dict(status="COMPLETE", qualified=False, scope="OWN_EARLY_MOVING_REFRESH_SAC_AUDIT",
        inherited_online_updates=report["inherited_online_updates"], restored_learner_bytes_identical=report.get("restored_learner_bytes_identical"),
        branch_audit=old, own_warmup_prefix_arrays_unchanged=17,
        warmup_goal_terminal_indices=goal_ends, final_goal_terminal_indices=expected_goal,
        goal_pool_min=min(pool_sizes), goal_pool_max=max(pool_sizes),
        declared_goal_slot_draws=goal_slot_count, total_draws=int(exposure.sum()),
        warmup_failure_terminal_indices=failure_ends, final_failure_terminal_indices=expected_failure,
        failure_pool_min=min(failure_pool_sizes), failure_pool_max=max(failure_pool_sizes),
        declared_failure_slot_draws=failure_slot_count,
        all_failure_terminal_draws=int(exposure[expected_failure].sum()),
        all_goal_terminal_draws=int(exposure[expected_goal].sum()),
        actual_transitions=count, actor_updates=count, critic_updates=count,
        initial_learner_sha256=initial["learner_sha256"],
        initial_reference_bytes_equal=initial["reference_bytes_equal"],
        original_checkpoint_sha256=sha(checkpoint / "manifest.json"),
        report_sha256=sha(directory / "report.json"), learner_sha256=sha(weights),
        replay_sha256=sha(directory / "replay.npz"), audit_sha256=sha(Path(__file__)),
        note="Own warmup and online lineage preserved; terminal labels verified against real outcome records. "
             "Every new transition has one full SAC update. Batch hashes/RNG reconstructed; "
             "distribution changed, no importance weights or production adoption claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "checkpoint", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.directory, args.checkpoint, args.manifest)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
