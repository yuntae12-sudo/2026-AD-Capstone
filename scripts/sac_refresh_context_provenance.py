"""Validate exact parent preservation for actual-context SAC continuation."""
import json
from pathlib import Path
import numpy as np


def validate_provenance(directory, manifest):
    from scripts.continue_sac_refresh_context import SCOPE, SOURCE_FILES, validate_budget
    from src.training.sac.checkpoint import file_hash
    directory, manifest = Path(directory), Path(manifest)
    read = lambda p: json.loads(Path(p).read_text())
    report = read(directory/"report.json")
    if report.get("status") != "COMPLETE" or report.get("scope") != SCOPE:
        raise ValueError("requires a completed actual-context continuation")
    history = report.get("continuation_history", [])
    if len(history) != 1:
        raise ValueError("only one verified fresh parent is supported")
    entry = history[0]
    if entry.get("method") != "replay_actual_context_without_collection_rng_restart":
        raise ValueError("collection restart cannot masquerade as actual-context resume")
    source_dir = Path(entry["source_directory"])
    if set(entry["source_hashes"]) != set(SOURCE_FILES):
        raise ValueError("incomplete parent hashes")
    for name, digest in entry["source_hashes"].items():
        if file_hash(source_dir/name) != digest:
            raise ValueError(f"parent checksum mismatch: {name}")
    source = read(source_dir/"report.json")
    contract = validate_budget(source, report["budget"])
    inherited = source["updates"]
    if (report["updates"] != report["budget"] or report["updates"] != report["new_transitions"]
            or report["inherited_online_updates"] != inherited
            or entry["inherited_online_updates"] != inherited or entry["source_scope"] != source["scope"]
            or report["collected_this_fork"] != report["updates"]-inherited
            or report.get("restored_learner_bytes_identical") is not True):
        raise ValueError("continuation counters or source identity mismatch")
    for key in ("versions", "source_versions", "seed", "collection_seed", "configuration_sha256",
                "configuration_overrides", "source_checkpoint_sha256", "inherited_self_warmup",
                "replay_sampling", contract):
        if report.get(key) != source.get(key):
            raise ValueError(f"continuation changed frozen setting: {key}")
    if file_hash(manifest) != report["versions"]["manifest_sha256"]:
        raise ValueError("manifest mismatch")
    if report["source_script_sha256"] != source["script_sha256"]:
        raise ValueError("parent driver identity mismatch")
    if report["script_sha256"] != file_hash(Path(__file__).with_name("continue_sac_refresh_context.py")):
        raise ValueError("continuation driver changed")
    for key, value in source["collection"].items():
        if not key.endswith("_transitions") and report["collection"].get(key) != value:
            raise ValueError(f"collection setting changed: {key}")
    for key, value in source["arms"]["online"].items():
        if key not in ("status", "updates", "learner_sha256") and report["arms"]["online"].get(key) != value:
            raise ValueError(f"learner setting changed: {key}")
    for name in ("ticks.jsonl", "updates.jsonl"):
        original = (source_dir/name).read_bytes().splitlines(keepends=True)
        current = (directory/name).read_bytes().splitlines(keepends=True)
        if len(original) != inherited or len(current) != report["updates"] or current[:inherited] != original:
            raise ValueError(f"parent raw history changed: {name}")
    for name in ("initial-learner.msgpack", "initial-state-audit.json"):
        if (source_dir/name).read_bytes() != (directory/name).read_bytes():
            raise ValueError("fresh origin evidence changed")
    rows = [json.loads(x) for x in (directory/"ticks.jsonl").read_text().splitlines()]
    parent_main = [r for r in rows[:inherited] if r["lineage"]["kind"] == "main"][-1]
    first = rows[inherited]
    if (first["lineage"]["kind"] != "main" or not parent_main["bootstrap"]
            or first["episode_uid"] != parent_main["episode_uid"]
            or first["pre_tick"] != parent_main["tick"]
            or any(first[k] != parent_main[k] for k in ("scene_id", "provider"))):
        raise ValueError("first resumed action is not the original main successor")
    restored = read(directory/"restored-state-audit.json")
    context = read(directory/"restored-context.json")
    if (restored["status"] != "COMPLETE" or restored["bytes_equal"] is not True
            or restored["source_learner_sha256"] != file_hash(source_dir/"online-learner.msgpack")
            or restored["restored_learner_sha256"] != restored["source_learner_sha256"]
            or restored["updates"] != inherited or restored["bc_updates"] != 0
            or any(restored[k] != inherited for k in ("actor_steps", "q1_steps", "q2_steps"))
            or restored["replay_size"] != 5000+inherited
            or restored["context_sha256"] != file_hash(directory/"restored-context.json")):
        raise ValueError("learner/replay restoration evidence mismatch")
    if (context["status"] != "COMPLETE" or context["actual_replayed_steps"] != inherited
            or context["main_episode_uid"] != parent_main["episode_uid"]
            or context["main_tick"] != parent_main["tick"]
            or context["policy_rng"] != restored["policy_rng"]
            or any(context[k] != 0 for k in ("new_policy_calls", "new_training_updates", "replay_insertions"))):
        raise ValueError("physical context restoration evidence mismatch")
    for path, digest in context["input_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError(f"restoration input changed: {path}")
    with np.load(source_dir/"replay.npz", allow_pickle=False) as original, np.load(directory/"replay.npz", allow_pickle=False) as current:
        fields = set(original.files)-{"metadata"}
        if len(fields) != 17 or fields != set(current.files)-{"metadata"}:
            raise ValueError("replay schema mismatch")
        for field in fields:
            if current[field][:5000+inherited].tobytes() != original[field].tobytes():
                raise ValueError(f"parent replay changed: {field}")
        parent_meta = json.loads(str(original["metadata"]))
        if restored["replay_rng"] != parent_meta["rng"]:
            raise ValueError("replay sampling RNG was restarted")
        # Same original main state must continue across interleaved refresh rows.
        j = parent_main["collection_id"]-1
        for field in ("obs", "fields", "items"):
            np.testing.assert_array_equal(current[field][5000+inherited], original["next_"+field][5000+j])
    return dict(status="COMPLETE", source_directory=str(source_dir), contract=contract,
        source_scope=source["scope"], inherited_online_updates=inherited,
        new_actual_transitions=report["updates"]-inherited, replay_prefix_fields=17,
        source_hashes=entry["source_hashes"], first_resumed_tick=first["pre_tick"],
        context_sha256=file_hash(directory/"restored-context.json"))
