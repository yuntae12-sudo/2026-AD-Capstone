#!/usr/bin/env python3
"""CPU-only audit of fixed-policy critic evaluation, not policy qualification."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(directory, source):
    import numpy as np
    from flax import serialization
    from scripts.sac_diagnostic_sampling import draw_indices, indices_hash, sampling_spec

    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    digest = lambda value: hashlib.sha256(value).hexdigest()
    report = read(directory / "report.json")
    original = read(source / "report.json")
    assert report["status"] == original["status"] == "COMPLETE"
    assert report["scope"] == "FIXED_POLICY_CRITIC_EVALUATION_NOT_SAC_TRAINING"
    assert report["qualified"] is False and report["production_checkpoint"] is False
    assert report["source_model_sha256"] == sha(source / "online-learner.msgpack")
    input_hashes = {Path(name).resolve(): value for name, value in report["input_sha256"].items()}
    assert input_hashes[(source / "report.json").resolve()] == sha(source / "report.json")
    assert input_hashes[(source / "replay.npz").resolve()] == sha(source / "replay.npz")
    for group in ("input_sha256", "source_sha256"):
        for name, expected in report[group].items():
            assert sha(Path(name)) == expected, name
    assert report["versions"] == original["versions"]
    assert report["replay_sampling"] == sampling_spec("half_recent256")
    budget, inherited = report["budget"], report["inherited_replay_size"]
    assert budget in (100, 1000) and inherited == 6000
    assert report["critic_steps"] == report["actual_transitions"] == budget
    assert report["actor_updates"] == report["temperature_updates"] == 0
    state_path, replay_path = directory / "critic-evaluation-state.msgpack", directory / "diagnostic-replay.npz"
    assert sha(state_path) == report["evaluation_state_sha256"]
    assert sha(replay_path) == report["diagnostic_replay_sha256"]
    before = serialization.msgpack_restore((source / "online-learner.msgpack").read_bytes())
    after = serialization.msgpack_restore(state_path.read_bytes())
    # Flax to_bytes normalizes state dictionaries; raw msgpack repacking alone
    # is not the same serialization path for nested optimizer state.
    for key in ("actor", "log_alpha", "alpha_optimizer"):
        assert serialization.to_bytes(before[key]) == serialization.to_bytes(after[key]), key
    assert digest(serialization.to_bytes(before["actor"])) == report["actor_sha256"]
    temperature = serialization.to_bytes((before["log_alpha"], before["alpha_optimizer"]))
    assert digest(temperature) == report["temperature_sha256"]
    assert int(after["updates"]) == int(before["updates"]) + budget
    assert int(after["bc_updates"]) == int(before["bc_updates"]) == 0
    critic_changed = {}
    for key in ("q1", "q2"):
        assert int(after[key]["step"]) == int(before[key]["step"]) + budget
        critic_changed[key] = serialization.to_bytes(after[key]["params"]) != serialization.to_bytes(before[key]["params"])
    ticks = [json.loads(line) for line in (directory / "ticks.jsonl").read_text().splitlines()]
    updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    assert len(ticks) == len(updates) == budget
    assert len(report["roots"]) == 2
    assert {r["provider"] for r in report["roots"]} == {"fixed_keep", "fixed_follow"}
    with np.load(source / "replay.npz", allow_pickle=False) as old, np.load(replay_path, allow_pickle=False) as new:
        names = set(old.files) - {"metadata"}
        assert len(names) == 17 and names == set(new.files) - {"metadata"}
        old_meta, meta = read_metadata(old), read_metadata(new)
        assert old_meta["size"] == old_meta["cursor"] == inherited
        assert meta["size"] == meta["cursor"] == inherited + budget < meta["capacity"]
        assert meta["versions"] == old_meta["versions"] == report["versions"]
        assert old_meta["rng"] == report["inherited_replay_rng"]
        assert meta["rng"] == report["final_replay_rng"]
        arrays = {}
        for key in names:
            value = new[key]
            assert value[:inherited].tobytes() == old[key].tobytes(), key
            assert digest(value[:inherited].tobytes()) == report["inherited_prefix_sha256"][key]
            arrays[key] = value[inherited:].copy()
        assert not arrays["teacher_accepted"].any() and not old["teacher_accepted"].any()
    rng = np.random.default_rng()
    rng.bit_generator.state = old_meta["rng"]
    counts = np.zeros(inherited + budget, dtype=np.int64)
    groups = {}

    def fingerprint(index, prefix=""):
        h = hashlib.sha256()
        for key in (prefix + "obs", prefix + "fields", prefix + "items"):
            array = arrays[key][index]
            h.update(str((array.dtype.str, array.shape)).encode())
            h.update(array.tobytes())
        return h.hexdigest()

    for i, (row, update) in enumerate(zip(ticks, updates)):
        count = i + 1
        assert row["collection_id"] == update["critic_steps"] == count
        assert update["actor_update_discarded"] and update["temperature_update_discarded"]
        metrics = update["canonical_hypothetical_metrics"]
        assert metrics["state_finite"] and all(math.isfinite(v) for v in metrics.values())
        assert update["replay_size"] == inherited + count
        indices = draw_indices(rng, inherited + count, "half_recent256")
        assert indices_hash(indices) == update["replay_indices_sha256"]
        np.add.at(counts, indices, 1)
        assert row["root_index"] == row["rollout"] % 2
        root = report["roots"][row["root_index"]]
        assert (row["scene_id"], row["provider"]) == (root["scene_id"], root["provider"])
        assert row["policy_seed"] == report["collection_seed"] + row["rollout"]
        assert row["pre_tick"] == root["tick"] + row["branch_step"]
        assert row["observation_sha256"] == fingerprint(i)
        assert row["next_observation_sha256"] == fingerprint(i, "next_")
        assert arrays["scene_id"][i] == row["scene_id"]
        assert arrays["episode_uid"][i] == row["episode_uid"]
        assert arrays["tick"][i] == row["pre_tick"]
        assert arrays["raw_u"].dtype == np.float64
        np.testing.assert_array_equal(arrays["raw_u"][i], row["raw_proposal"]["raw_u"])
        for key in ("reward", "bootstrap", "delta_ticks", "terminated", "truncated"):
            assert arrays[key][i] == row[key], key
        assert row["delta_ticks"] == 1
        assert row["bootstrap"] == (not (row["terminated"] or row["truncated"]))
        group = groups.setdefault(row["rollout"], [])
        assert row["branch_step"] == len(group)
        if group:
            assert group[-1]["bootstrap"]
            assert group[-1]["next_observation_sha256"] == row["observation_sha256"]
        else:
            assert row["observation_sha256"] == root["observation_sha256"]
        group.append(row)
    assert rng.bit_generator.state == meta["rng"]
    assert list(groups) == list(range(len(groups)))
    assert len(groups) == len(report["episodes"])
    for episode in report["episodes"]:
        rows = groups[episode["rollout"]]
        assert episode["ticks"] == len(rows)
        assert episode["accepted"] == sum(r["check_valid"] for r in rows)
        assert math.isclose(episode["return_training"], math.fsum(r["reward"] for r in rows), abs_tol=1e-10, rel_tol=0)
        if rows[-1]["bootstrap"]:
            assert rows[-1]["collection_id"] == budget and episode["outcome"] == "DIAGNOSTIC_BUDGET"
        else:
            assert episode["outcome"] == rows[-1]["outcome"]
    assert [p["critic_steps"] for p in report["probes"]] == list(range(0, budget + 1, 100))
    for probe in report["probes"]:
        assert len(probe["cases"]) == 2
        for row, root in zip(probe["cases"], report["roots"]):
            assert row["provider"] == root["provider"] and row["scene_id"] == root["scene_id"]
            assert all(math.isfinite(row[k]) for k in ("q", "target_q", "q_minus_reference_mean"))
            assert math.isclose(row["q_minus_reference_mean"], row["q"]-root["reference_summary"]["mean_soft_return"], abs_tol=1e-10, rel_tol=0)
    paths = [directory / "report.json", state_path, replay_path, directory / "ticks.jsonl",
             directory / "updates.jsonl", source / "report.json", source / "online-learner.msgpack", source / "replay.npz"]
    return dict(status="COMPLETE", qualified=False, scope="FIXED_POLICY_CRITIC_AUDIT_NOT_POLICY_QUALIFICATION",
                actual_transitions=budget, critic_steps=budget, complete_rollouts=sum(not rows[-1]["bootstrap"] for rows in groups.values()),
                inherited_arrays_bytes_identical=17, actor_and_temperature_unchanged=True,
                critic_parameters_changed=critic_changed,
                actual_transition_chains_verified=True, every_sampling_hash_and_final_rng_verified=True,
                draws=dict(total=int(counts.sum()), inherited=int(counts[:inherited].sum()),
                           new=int(counts[inherited:].sum()), new_never_drawn=int((counts[inherited:] == 0).sum())),
                probes=report["probes"], source_sha256={str(p): sha(p) for p in paths},
                reviewer_sha256=sha(Path(__file__)),
                note="Read-only replay/serialization/provenance/chain/sampling audit. No new simulator re-execution or critic forward pass. "
                     "Actor and temperature are unchanged; this cannot be evidence of better driving. Two train roots and four MC reference samples/root are not generalization evidence.")


def read_metadata(archive):
    return json.loads(str(archive["metadata"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.diagnostic, args.source)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "actual_transitions", "complete_rollouts", "draws")}))


if __name__ == "__main__":
    main()
