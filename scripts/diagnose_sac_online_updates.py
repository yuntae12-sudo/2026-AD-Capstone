#!/usr/bin/env python3
"""Fresh short online diagnostic with inherited self warmup, not a production run."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "configuration", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument(
        "--mean-gain",
        type=float,
        help="Explicit diagnostic override of the approved linear mean gain",
    )
    parser.add_argument(
        "--save-replay",
        action="store_true",
        help="Persist the actual replay for read-only coverage audits",
    )
    args = parser.parse_args()
    if not 1 <= args.updates <= 1000:
        parser.error("short diagnostic budget must be 1..1000 updates")
    if args.mean_gain is not None and (
        not math.isfinite(args.mean_gain) or args.mean_gain <= 0
    ):
        parser.error("mean gain must be finite and positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import numpy as np
    from flax import serialization
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.networks import Actor, Critic
    from src.training.sac.learner import initialize, update
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import Replay
    from src.training.sac.collector import collect_step
    from src.contracts.observation import CoreObservation
    from src.contracts.planning import to_wire
    from src.scenarios.womd20 import parse_scenario
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from scripts.sac_experiment_runtime import model_proposer

    config = json.loads((args.configuration / "report.json").read_text())
    variant = dict(config["arms"][args.variant])
    overrides = {} if args.mean_gain is None else dict(mean_gain=args.mean_gain)
    variant.update(overrides)
    version = versions(args.manifest)
    if config["versions"] != version or variant["status"] != "COMPLETE":
        raise ValueError("requires a completed matching-runtime configuration")
    seed = config["seed"]
    meta = json.loads((args.checkpoint / "manifest.json").read_text())
    origin = meta["run_state"]["report"]
    if not (
        meta["run_state"]["seed"] == seed
        and origin["training_mode"] == "pure-sac"
        and origin["warmup_transitions"] == 5000
        and origin["bc_checkpoint_sha256"] is None
        and origin["initial_replay_size"] == 0
    ):
        raise ValueError("requires same-seed pure self warmup")
    if (
        meta["versions"]["manifest_sha256"] != version["manifest_sha256"]
        or meta["versions"]["contract_sha256"] != version["contract_sha256"]
    ):
        raise ValueError("warmup data/contract mismatch")
    if file_hash(args.checkpoint / "replay.npz") != meta["hashes"]["replay.npz"]:
        raise ValueError("replay checksum mismatch")
    replay = Replay.load(args.checkpoint / "replay.npz", meta["versions"])
    if (
        not 5000 <= replay.size < replay.capacity
        or replay.cursor != replay.size
        or np.any(replay.arrays["teacher_accepted"][:5000])
    ):
        raise ValueError("warmup prefix unavailable or contains teacher data")
    replay.size = replay.cursor = 5000
    replay.rng = np.random.default_rng(seed)
    # Explicit mixed-lineage diagnostic; source version is retained below.
    replay.versions = version
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    with jax.default_device(gpu):
        state = initialize(
            tensor_observation(CoreObservation.empty()),
            seed,
            actor=Actor(
                mean_gain=variant["mean_gain"],
                mean_basis=variant["mean_basis"],
                head_layer_norm=variant["actor_head_layer_norm"],
                mean_coordinate_scale=variant.get("mean_coordinate_scale", "raw"),
            ),
            critic=Critic(
                zero_output=variant["zero_q"],
                action_coordinates=variant.get("critic_action_coordinates", "raw"),
            ),
        )
    policy, proposer = model_proposer(state.actor, seed, False, gpu)
    report = dict(
        status="RUNNING",
        scope="SHORT_ONLINE_SAC_DIAGNOSTIC",
        qualified=False,
        versions=version,
        source_versions=meta["versions"],
        seed=seed,
        source_checkpoint_sha256=file_hash(args.checkpoint / "manifest.json"),
        configuration_sha256=file_hash(args.configuration / "report.json"),
        configuration_overrides=overrides,
        script_sha256=file_hash(__file__),
        inherited_self_warmup=5000,
        new_transitions=0,
        updates=0,
        budget=args.updates,
        note="Fresh learner; first 5000 old own-policy warmup only. Scene/policy RNG restart: a new diagnostic branch, not exact continuation. Each new transition is inserted before one update. Validation never enters replay. Mid-episode budget endings are not successful episodes.",
        episodes=[],
        arms={},
    )
    args.output.mkdir(parents=True)
    candidates = selected_scenes(args.manifest, "train")
    rng = np.random.default_rng(seed)
    try:
        while int(state.updates) < args.updates:
            for index in rng.permutation(len(candidates)):
                row, path = candidates[index]
                with jax.default_device(cpu):
                    env = SACPlanningEnvironment()
                    boundary = env.reset(
                        parse_scenario(path.read_bytes()),
                        seed=seed,
                        provider_spec=row["provider"],
                    )
                episode = dict(
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    ticks=0,
                    accepted=0,
                    return_training=0.0,
                    outcome="IN_PROGRESS",
                    reasons={},
                )
                report["episodes"].append(episode)
                while int(state.updates) < args.updates:
                    policy.actor = state.actor
                    with jax.default_device(cpu):
                        step = collect_step(env, boundary, proposer, replay)
                    boundary = step.boundary
                    result = step.result
                    if step.transition is None:
                        raise RuntimeError("invalid diagnostic transition; stop")
                    with jax.default_device(gpu):
                        state, metrics = update(state, replay.sample(256), replay.size)
                    episode["ticks"] += 1
                    episode["accepted"] += int(result.check_result.valid)
                    episode["return_training"] += step.reward.training
                    reason = result.check_result.reason
                    episode["reasons"][reason] = episode["reasons"].get(reason, 0) + 1
                    report["new_transitions"] += 1
                    report["updates"] = int(state.updates)
                    with (args.output / "ticks.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                dict(
                                    update=int(state.updates),
                                    scene_id=row["scene_id"],
                                    provider=row["provider"],
                                    tick=env.tick,
                                    raw_proposal=to_wire(result.raw_proposal),
                                    check_valid=result.check_result.valid,
                                    check_reason=reason,
                                    violations=list(result.check_result.violations),
                                    source=result.source.value,
                                    actual_control=to_wire(result.actual_control),
                                    reward=step.reward.training,
                                    terminated=result.terminated,
                                    truncated=result.truncated,
                                ),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    with (args.output / "updates.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                dict(
                                    update=int(state.updates),
                                    **{k: float(v) for k, v in metrics.items()},
                                ),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    if result.terminated or result.truncated:
                        episode["outcome"] = result.terminal_reason.value
                    write_report(args.output / "report.json", report)
                    if result.terminated or result.truncated:
                        break
                if episode["outcome"] == "IN_PROGRESS":
                    episode["outcome"] = "DIAGNOSTIC_BUDGET"
                print(json.dumps(episode), flush=True)
                if int(state.updates) >= args.updates:
                    break
        if versions(args.manifest) != version:
            raise RuntimeError("runtime changed during online diagnostic")
        weights = args.output / "online-learner.msgpack"
        weights.write_bytes(serialization.to_bytes(state))
        report["arms"]["online"] = dict(
            status="COMPLETE",
            mean_gain=variant["mean_gain"],
            mean_basis=variant["mean_basis"],
            mean_coordinate_scale=variant.get("mean_coordinate_scale", "raw"),
            actor_head_layer_norm=variant["actor_head_layer_norm"],
            zero_q=variant["zero_q"],
            critic_action_coordinates=variant.get("critic_action_coordinates", "raw"),
            updates=int(state.updates),
            learner_sha256=file_hash(weights),
        )
        if args.save_replay:
            recorded = [
                json.loads(line)["raw_proposal"]["raw_u"]
                for line in (args.output / "ticks.jsonl").read_text().splitlines()
            ]
            np.testing.assert_array_equal(
                replay.arrays["raw_u"][5000 : replay.size],
                np.asarray(recorded, np.float64),
            )
            if replay.size != 5000 + report["new_transitions"] or np.any(
                replay.arrays["teacher_accepted"][: replay.size]
            ):
                raise ValueError("unexpected diagnostic replay composition")
            replay.save(args.output / "replay.npz")
            report["saved_replay"] = dict(
                file="replay.npz",
                sha256=file_hash(args.output / "replay.npz"),
                size=replay.size,
                own_warmup=5000,
                new_online=report["new_transitions"],
            )
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
