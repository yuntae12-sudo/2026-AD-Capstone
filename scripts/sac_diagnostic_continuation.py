"""Validate a bounded own-policy diagnostic continuation without changing SAC."""

import json
from pathlib import Path

from src.training.sac.checkpoint import file_hash


ARM_FIELDS = (
    "mean_gain", "mean_basis", "mean_coordinate_scale", "actor_head_layer_norm",
    "zero_q", "critic_action_coordinates", "critic_action_kernel_init",
    "critic_action_gain", "critic_factory_sha256",
)


def validate_source(directory, version, variant, collection, total_budget):
    """Check lineage before restoring optimizer/target/RNG and actual replay."""
    directory = Path(directory)
    source = json.loads((directory / "report.json").read_text())
    arm = source.get("arms", {}).get("online", {})
    updates = source.get("updates", 0)
    if (source.get("status") != "COMPLETE" or arm.get("status") != "COMPLETE"
            or source.get("versions") != version
            or source.get("inherited_self_warmup") != 5000
            or updates != arm.get("updates")
            or updates != source.get("new_transitions")
            or not 0 < updates < total_budget <= 5000):
        raise ValueError("invalid completed diagnostic or bounded continuation budget")
    for key in ARM_FIELDS:
        if arm.get(key) != variant.get(key):
            raise ValueError(f"continuation cannot change model setting: {key}")
    for key in ("mode", "samples", "window", "selection", "branch_script_sha256", "fork_script_sha256"):
        if source["collection"].get(key) != collection.get(key):
            raise ValueError(f"continuation cannot change collection setting: {key}")
    for key, default in (("rollout_horizon", 1), ("rollout_script_sha256", None), ("restart_terminal", False), ("complete_path", False)):
        if source["collection"].get(key, default) != collection.get(key, default):
            raise ValueError(f"continuation cannot change collection setting: {key}")
    saved = source.get("saved_replay", {})
    if (saved.get("size") != 5000 + updates or saved.get("own_warmup") != 5000
            or saved.get("new_online") != updates):
        raise ValueError("incomplete replay lineage")
    expected = {"online-learner.msgpack": arm["learner_sha256"],
                "replay.npz": saved["sha256"]}
    for name, digest in expected.items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"continuation checksum mismatch: {name}")
    for filename in ("ticks.jsonl", "updates.jsonl"):
        rows = [json.loads(line) for line in (directory / filename).read_text().splitlines()]
        if len(rows) != updates or [r["update"] for r in rows] != list(range(1, updates + 1)):
            raise ValueError(f"incomplete continuation history: {filename}")
    return source


def continuation_lineage(directory, source, collection_seed):
    directory = Path(directory)
    return dict(
        source_directory=str(directory.resolve()),
        source_hashes={name: file_hash(directory / name) for name in (
            "report.json", "online-learner.msgpack", "replay.npz", "ticks.jsonl", "updates.jsonl")},
        inherited_online_updates=source["updates"],
        collection_seed=collection_seed,
        note="Learner/optimizer/targets/learner RNG and uniform replay RNG continue. "
             "Collection environment and scene/policy RNG restart explicitly; not exact rollout resume. "
             "No artificial terminal is inserted for the inherited budget boundary.",
    )
