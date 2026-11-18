#!/usr/bin/env python3
"""Closed-loop validation of explicitly configured diagnostic SAC weights.

No replay insertion or training. Results remain diagnostic, not a production gate.
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument(
        "--scenes", nargs="+", help="Explicit validation scene IDs; omit for all"
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    from flax import serialization
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.networks import Actor, Critic
    from src.training.sac.learner import initialize
    from src.training.sac.policy import tensor_observation
    from src.contracts.observation import CoreObservation
    from src.scenarios.womd20 import parse_scenario
    from scripts.sac_experiment_runtime import model_proposer
    from src.visualization.sac_rollout import capture

    source = json.loads((args.diagnostic / "report.json").read_text())
    arm = source["arms"][args.variant]
    version = versions(args.manifest)
    if arm["status"] != "COMPLETE" or source["versions"] != version:
        raise ValueError("requires completed, identical-runtime diagnostic arm")
    weights = args.diagnostic / f"{args.variant}-learner.msgpack"
    if arm.get("learner_sha256") and file_hash(weights) != arm["learner_sha256"]:
        raise ValueError("diagnostic weight checksum mismatch")
    seed = source["seed"]
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    with jax.default_device(gpu):
        initial = initialize(
            tensor_observation(CoreObservation.empty()),
            seed,
            actor=Actor(
                mean_gain=arm["mean_gain"],
                mean_basis=arm.get("mean_basis", "direct"),
                head_layer_norm=arm.get("actor_head_layer_norm", False),
            ),
            critic=Critic(
                zero_output=arm["zero_q"],
                action_coordinates=arm.get("critic_action_coordinates", "raw"),
            ),
        )
        trained = serialization.from_bytes(initial, weights.read_bytes())
        trained = jax.device_put(trained, gpu)
    if int(trained.updates) != arm["updates"] or int(trained.bc_updates) != 0:
        raise ValueError("diagnostic learner counter mismatch")
    selected = selected_scenes(args.manifest, "validation")
    if args.scenes:
        if not set(args.scenes) <= {r["scene_id"] for r, _ in selected}:
            raise ValueError("scene absent from validation")
        selected = [(r, path) for r, path in selected if r["scene_id"] in args.scenes]
    args.output.mkdir(parents=True)
    report = dict(
        status="RUNNING",
        scope="DIAGNOSTIC_PAIRED_CLOSED_LOOP",
        qualified=False,
        versions=version,
        diagnostic_sha256=file_hash(args.diagnostic / "report.json"),
        learner_sha256=file_hash(weights),
        script_sha256=file_hash(__file__),
        seed=seed,
        variant=args.variant,
        mean_gain=arm["mean_gain"],
        zero_q=arm["zero_q"],
        mean_basis=arm.get("mean_basis", "direct"),
        actor_head_layer_norm=arm.get("actor_head_layer_norm", False),
        critic_action_coordinates=arm.get("critic_action_coordinates", "raw"),
        selected_pairs=[(r["scene_id"], r["provider"]) for r, _ in selected],
        note="Paired untrained and short-update deterministic mean, CPU Waymax/GPU inference. Selected scenes are not independent test. No teacher or BC.",
        episodes=[],
    )
    write_report(args.output / "report.json", report)
    for label, state in (("untrained", initial), ("updated", trained)):
        _, proposer = model_proposer(state.actor, seed, True, gpu)
        for row, path in selected:
            with jax.default_device(cpu):
                episode = capture(
                    parse_scenario(path.read_bytes()),
                    row["provider"],
                    proposer,
                    label,
                    seed,
                )
            name = f"{label}-{row['scene_id']}-{row['provider']}.json"
            write_report(args.output / name, episode)
            summary = dict(
                label=label,
                scene_id=row["scene_id"],
                provider=row["provider"],
                outcome=episode["outcome"],
                **episode["summary"],
                evidence=name,
            )
            report["episodes"].append(summary)
            write_report(args.output / "report.json", report)
            print(json.dumps(summary), flush=True)
    if versions(args.manifest) != version:
        raise RuntimeError("runtime changed during diagnostic rollout")
    report["status"] = "COMPLETE"
    write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
