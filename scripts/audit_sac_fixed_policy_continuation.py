#!/usr/bin/env python3
"""Audit frozen-policy diagnostic continuation, including inherited cutoffs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(directory, parent_directory, source_directory):
    import numpy as np
    from flax import serialization
    from scripts.audit_sac_fixed_policy_critic import audit as audit_parent
    from scripts.sac_diagnostic_sampling import draw_indices, indices_hash, sampling_spec
    from scripts.sac_fixed_policy_continuation import continuation_plan

    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    digest = lambda b: hashlib.sha256(b).hexdigest()
    audit_parent(parent_directory, source_directory)
    report, parent = (read(p / "report.json") for p in (directory, parent_directory))
    assert report["status"] == "COMPLETE"
    assert report["scope"] == "FIXED_POLICY_CRITIC_CONTINUATION_NOT_SAC_TRAINING"
    assert not report["qualified"] and not report["production_checkpoint"]
    assert Path(report["parent_directory"]).resolve() == parent_directory.resolve()
    assert Path(report["source_directory"]).resolve() == source_directory.resolve()
    for group in ("source_sha256", "input_sha256"):
        for p, expected in report[group].items():
            assert sha(Path(p)) == expected, p
    inputs = {Path(p).resolve(): h for p, h in report["input_sha256"].items()}
    for name in ("report.json", "critic-evaluation-state.msgpack", "diagnostic-replay.npz", "ticks.jsonl", "updates.jsonl"):
        p = parent_directory / name
        assert inputs[p.resolve()] == sha(p)
    assert report["source_model_sha256"] == parent["source_model_sha256"]
    assert report["versions"] == parent["versions"]
    assert report["replay_sampling"] == parent["replay_sampling"] == sampling_spec("half_recent256")
    old_ticks = (parent_directory / "ticks.jsonl").read_bytes()
    old_updates = (parent_directory / "updates.jsonl").read_bytes()
    for name, prefix in (("ticks", old_ticks), ("updates", old_updates)):
        assert (directory / (name + ".jsonl")).read_bytes().startswith(prefix), name
        assert len(prefix) == report["inherited_" + name + "_bytes"]
        assert digest(prefix) == report["inherited_" + name + "_sha256"]
    ticks = [json.loads(line) for line in (directory / "ticks.jsonl").read_text().splitlines()]
    updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    total = report["budget"]
    assert report["critic_steps"] == report["actual_transitions"] == len(ticks) == len(updates) == total
    assert report["new_critic_steps"] == report["new_actual_transitions"] == total - 1000
    assert report["actor_updates"] == report["temperature_updates"] == 0
    assert report["roots"] == parent["roots"]
    assert report["episodes"][:len(parent["episodes"])] == parent["episodes"]
    state_path = directory / "critic-evaluation-state.msgpack"
    replay_path = directory / "diagnostic-replay.npz"
    assert sha(state_path) == report["evaluation_state_sha256"]
    assert sha(replay_path) == report["diagnostic_replay_sha256"]
    assert sha(parent_directory / state_path.name) == report["inherited_state_sha256"]
    before = serialization.msgpack_restore((parent_directory / state_path.name).read_bytes())
    after = serialization.msgpack_restore(state_path.read_bytes())

    def check_state(value, added):
        for key in ("actor", "log_alpha", "alpha_optimizer"):
            assert serialization.to_bytes(value[key]) == serialization.to_bytes(before[key]), key
        assert int(value["updates"]) == int(before["updates"]) + added
        assert int(value["bc_updates"]) == int(before["bc_updates"]) == 0
        for key in ("q1", "q2"):
            assert int(value[key]["step"]) == int(before[key]["step"]) + added

    check_state(after, total - 1000)
    assert digest(serialization.to_bytes(before["actor"])) == report["actor_sha256"] == parent["actor_sha256"]
    assert digest(serialization.to_bytes((before["log_alpha"], before["alpha_optimizer"]))) == report["temperature_sha256"]
    milestones = sorted(set(range(2000, total + 1, 1000)) | {total})
    assert [s["critic_steps"] for s in report["snapshots"]] == milestones
    for snapshot in report["snapshots"]:
        path = directory / snapshot["state_file"]
        assert path.resolve().parent == directory.resolve()
        assert sha(path) == snapshot["sha256"]
        check_state(serialization.msgpack_restore(path.read_bytes()), snapshot["critic_steps"] - 1000)
    assert report["snapshots"][-1]["sha256"] == sha(state_path)
    with np.load(parent_directory / "diagnostic-replay.npz", allow_pickle=False) as old, np.load(replay_path, allow_pickle=False) as new:
        old_meta, meta = (json.loads(str(z["metadata"])) for z in (old, new))
        plan = continuation_plan(parent, json.loads(old_ticks.splitlines()[-1]), old_meta, total)
        assert report["plan"] == plan
        assert report["inherited_replay_size"] == old_meta["size"] == 7000
        assert meta["size"] == meta["cursor"] == report["final_replay_size"] == plan["final_replay_size"]
        assert meta["capacity"] == old_meta["capacity"] and meta["size"] < meta["capacity"]
        assert meta["versions"] == old_meta["versions"] == report["versions"]
        assert report["inherited_replay_rng"] == old_meta["rng"]
        assert report["final_replay_rng"] == meta["rng"]
        names = set(old.files) - {"metadata"}
        assert len(names) == 17 and names == set(new.files) - {"metadata"}
        arrays = {}
        for key in names:
            value = new[key]
            assert value[:7000].tobytes() == old[key].tobytes(), key
            assert digest(value[:7000].tobytes()) == report["inherited_prefix_sha256"][key]
            arrays[key] = value[7000:].copy()
        assert not old["teacher_accepted"].any() and not arrays["teacher_accepted"].any()
    rng = np.random.default_rng()
    rng.bit_generator.state = old_meta["rng"]
    counts = np.zeros(plan["final_replay_size"], dtype=np.int64)
    groups = {}
    for row in ticks:
        group = groups.setdefault(row["rollout"], [])
        assert row["branch_step"] == len(group)
        if group:
            assert group[-1]["bootstrap"]
            assert group[-1]["next_observation_sha256"] == row["observation_sha256"]
        group.append(row)

    def fingerprint(i, prefix=""):
        h = hashlib.sha256()
        for key in (prefix + "obs", prefix + "fields", prefix + "items"):
            value = arrays[key][i]
            h.update(str((value.dtype.str, value.shape)).encode())
            h.update(value.tobytes())
        return h.hexdigest()

    for i, (row, update) in enumerate(zip(ticks[1000:], updates[1000:])):
        count = i + 1001
        assert row["collection_id"] == update["critic_steps"] == count
        assert update["replay_size"] == 6000 + count
        assert update["actor_update_discarded"] and update["temperature_update_discarded"]
        metrics = update["canonical_hypothetical_metrics"]
        assert metrics["state_finite"] and all(math.isfinite(v) for v in metrics.values())
        indices = draw_indices(rng, 6000 + count, "half_recent256")
        assert indices_hash(indices) == update["replay_indices_sha256"]
        np.add.at(counts, indices, 1)
        assert row["rollout"] >= plan["first_new_rollout"]
        assert row["root_index"] == row["rollout"] % 2
        root = report["roots"][row["root_index"]]
        assert (row["scene_id"], row["provider"]) == (root["scene_id"], root["provider"])
        assert row["policy_seed"] == plan["collection_seed"] + row["rollout"]
        assert row["pre_tick"] == root["tick"] + row["branch_step"]
        assert row["observation_sha256"] == fingerprint(i)
        assert row["next_observation_sha256"] == fingerprint(i, "next_")
        if row["branch_step"] == 0:
            assert row["observation_sha256"] == root["observation_sha256"]
        for key in ("scene_id", "episode_uid", "reward", "bootstrap", "delta_ticks", "terminated", "truncated"):
            assert arrays[key][i] == row[key], key
        assert arrays["tick"][i] == row["pre_tick"]
        assert arrays["raw_u"].dtype == np.float64
        np.testing.assert_array_equal(arrays["raw_u"][i], row["raw_proposal"]["raw_u"])
        assert row["delta_ticks"] == 1
        assert row["bootstrap"] == (not (row["terminated"] or row["truncated"]))
    assert rng.bit_generator.state == meta["rng"]
    assert list(groups) == list(range(len(report["episodes"])))
    preserved_cutoffs = []
    for episode in report["episodes"]:
        rows = groups[episode["rollout"]]
        assert episode["ticks"] == len(rows)
        assert episode["accepted"] == sum(r["check_valid"] for r in rows)
        assert math.isclose(episode["return_training"], math.fsum(r["reward"] for r in rows), abs_tol=1e-10, rel_tol=0)
        if rows[-1]["bootstrap"]:
            assert rows[-1]["collection_id"] in (1000, total)
            assert episode["outcome"] == "DIAGNOSTIC_BUDGET"
            preserved_cutoffs.append(rows[-1]["collection_id"])
        else:
            assert episode["outcome"] == rows[-1]["outcome"]
    assert report["probes"][:len(parent["probes"])] == parent["probes"]
    assert [p["critic_steps"] for p in report["probes"]] == list(range(0, total+1, 100))
    assert report["resume_probe"]["critic_steps"] == 1000 and report["resume_probe_tolerance"] == 1e-3
    for probe in report["probes"] + [report["resume_probe"]]:
        assert len(probe["cases"]) == 2
        for row, root in zip(probe["cases"], report["roots"]):
            assert row["provider"] == root["provider"] and row["scene_id"] == root["scene_id"]
            assert all(math.isfinite(row[k]) for k in ("q", "target_q", "q_minus_reference_mean"))
            assert math.isclose(row["q_minus_reference_mean"], row["q"]-root["reference_summary"]["mean_soft_return"], abs_tol=1e-10, rel_tol=0)
    for new, old in zip(report["resume_probe"]["cases"], parent["probes"][-1]["cases"]):
        for key in ("q", "target_q"):
            assert abs(new[key]-old[key]) <= 1e-3
    paths = [directory / name for name in ("report.json", "ticks.jsonl", "updates.jsonl", "critic-evaluation-state.msgpack", "diagnostic-replay.npz")]
    paths += [directory / s["state_file"] for s in report["snapshots"]]
    return dict(status="COMPLETE", qualified=False, scope="FIXED_POLICY_CONTINUATION_AUDIT",
                total_critic_steps=total, new_actual_transitions=total-1000,
                inherited_arrays_bytes_identical=17, inherited_trace_bytes_identical=True,
                actor_and_temperature_unchanged=True, all_new_sampling_hashes_and_final_rng_verified=True,
                inherited_budget_bootstrap_preserved=True,
                inherited_boundary_bootstrap=plan["inherited_boundary"]["bootstrap"],
                budget_cutoff_collection_ids=preserved_cutoffs, snapshots_verified=len(report["snapshots"]),
                new_draws=dict(total=int(counts.sum()), inherited=int(counts[:7000].sum()), new=int(counts[7000:].sum())),
                complete_rollouts=sum(not rows[-1]["bootstrap"] for rows in groups.values()),
                source_sha256={str(p): sha(p) for p in paths}, reviewer_sha256=sha(Path(__file__)),
                note="Read-only continuation/replay/serialization/sampling audit, not simulator re-execution or policy performance. "
                     "Inherited and final budget cutoffs retain their true bootstrap. Actor/alpha fixed; no policy promotion.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "parent", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.diagnostic, args.parent, args.source)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "total_critic_steps", "new_actual_transitions", "new_draws")}))


if __name__ == "__main__":
    main()
