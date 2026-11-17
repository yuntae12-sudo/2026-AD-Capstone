#!/usr/bin/env python3
"""Fork unchanged self-collected warmup with freshly initialized learner state.

Only a zero-update, zero-BC same-seed boundary is eligible. This is an explicit
new experiment fork, not a claim of exact continuation of the old learner.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "synthetic", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import jax, numpy as np
    from flax import serialization
    from src.contracts.observation import CoreObservation
    from src.training.sac.learner import initialize
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import Replay
    from src.training.sac.checkpoint import file_hash, save_checkpoint
    from src.training.sac.runtime import versions, write_report
    from scripts.sac_experiment_runtime import checkpoint_metadata, runner_hash
    from scripts.pure_sac_protocol import (
        protocol_metadata,
        validate_pure_state,
        require_synthetic_report,
    )

    checkpoint = args.checkpoint.resolve()
    old = json.loads((checkpoint / "manifest.json").read_text())
    checkpoint_metadata(checkpoint, old["versions"])
    run = copy.deepcopy(old["run_state"])
    report = run["report"]
    count = report["valid_online_transitions"]
    seed = run["seed"]
    if not (
        0 < count < 5000
        and report["training_mode"] == "pure-sac"
        and report["updates"] == 0
        and report["initial_replay_size"] == 0
        and report["bc_checkpoint_sha256"] is None
        and run["metrics_bytes"] == 0
    ):
        raise ValueError("only self-collected pre-update warmup can be forked")
    synthetic = json.loads(args.synthetic.read_text())
    version = versions(args.manifest)
    reuse = synthetic.get("evidence_reuse", {})
    if (
        reuse.get("source_versions") != old["versions"]
        or reuse.get("method") != "verified_unchanged_execution_dependencies"
    ):
        raise ValueError("requires dependency-verified source compatibility evidence")
    # Revalidate compatibility now, rather than trusting a manually edited envelope.
    from scripts.revalidate_sac_evidence import (
        require_unchanged_dependencies,
        FOLDERS,
        ROOT,
    )
    import subprocess

    revision = reuse["compatibility_revision"]
    names = [
        str(path.relative_to(ROOT))
        for folder in FOLDERS
        for path in (ROOT / folder).glob("*.py")
    ]
    before = {
        name: subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
        for name in names
    }
    after = {name: (ROOT / name).read_bytes() for name in names}
    require_unchanged_dependencies(before, after)
    require_synthetic_report(
        synthetic, version, file_hash(ROOT / "scripts/validate_sac_exploration.py")
    )
    raw = serialization.msgpack_restore((checkpoint / "learner.msgpack").read_bytes())
    if int(raw["updates"]) != 0 or int(raw["bc_updates"]) != 0:
        raise ValueError("source contains learned weights")
    state = initialize(tensor_observation(CoreObservation.empty()), seed)
    old_actor = serialization.from_state_dict(
        state.actor.params, raw["actor"]["params"]
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(state.actor.params),
        jax.tree_util.tree_leaves(old_actor),
    ):
        np.testing.assert_array_equal(a, b)
    replay = Replay.load(checkpoint / "replay.npz", old["versions"])
    validate_pure_state(state, replay, count)
    args.output.mkdir(parents=True)
    # Copy precisely the checkpoint's committed trace prefix, not later failures.
    source = checkpoint.parent.parent / "ticks.jsonl"
    remaining = run["trace_bytes"]
    with source.open("rb") as src, (args.output / "ticks.jsonl").open("xb") as dst:
        while remaining:
            chunk = src.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("source committed trace is incomplete")
            dst.write(chunk)
            remaining -= len(chunk)
    valid = 0
    last = None
    with (args.output / "ticks.jsonl").open() as stream:
        for line in stream:
            last = json.loads(line)
            valid += int(last["sample_valid"])
    if valid != count or not (last["terminated"] or last["truncated"]):
        raise ValueError("warmup trace count/boundary mismatch")
    provenance = dict(
        source_checkpoint_sha256=file_hash(checkpoint / "manifest.json"),
        source_versions=old["versions"],
        inherited_self_collected_transitions=count,
        seed=seed,
        source_updates=0,
        source_bc_updates=0,
        source_trace_sha256=file_hash(args.output / "ticks.jsonl"),
        fresh_learner=True,
        note="New critic/optimizers/targets; fresh actor equals source untrained actor exactly. Scene and policy RNG retain same-seed warmup lineage.",
    )
    replay.versions = version
    report.update(
        status="READY_TO_RESUME",
        versions=version,
        runner_sha256=runner_hash(),
        protocol=protocol_metadata(),
        synthetic_sha256=file_hash(args.synthetic),
        warmup_reuse=provenance,
        updates=0,
        alpha_bound_updates=0,
        devices=dict(
            simulator=str(jax.devices("cpu")[0]), learner=str(jax.devices()[0])
        ),
    )
    destination = args.output / "checkpoints" / f"warmup-fork-{count:08d}"
    save_checkpoint(destination, state, replay, version, run, episode_boundary=True)
    write_report(
        args.output / "latest_recovery.json",
        dict(path=str(destination.relative_to(args.output))),
    )
    write_report(args.output / "report.json", report)
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
