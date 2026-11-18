#!/usr/bin/env python3
"""Paired short SAC updates on frozen self-collected warmup, never a training gate.

Defaults preserve production behavior. Explicit variants alter mean-head gain or
critic initialization; the objective, sampler, checker and PD stay unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VARIANTS = {
    "original": (1.0, False, "direct", False),
    "zero_q": (1.0, True, "direct", False),
    "gain4": (1e-4, False, "direct", False),
    "gain4_zero": (1e-4, True, "direct", False),
    "gain5_zero": (1e-5, True, "direct", False),
    "gain6_zero": (1e-6, True, "direct", False),
    "basis": (1.0, False, "integrated", False),
    "basis_zero": (1.0, True, "integrated", False),
    "basis01_zero": (0.1, True, "integrated", False),
    "basis_norm_zero": (1.0, True, "integrated", True),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.updates < 100:
        p.error("at least 100 updates required")
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import numpy as np
    from flax import serialization
    from src.contracts.observation import CoreObservation
    from src.training.sac.networks import (
        Actor,
        Critic,
        sample_action,
        observation_speed,
    )
    from src.training.sac.policy import tensor_observation, proposal_from_action
    from src.training.sac.replay import Replay
    from src.training.sac.learner import initialize, update
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from src.scenarios.womd20 import parse_scenario
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.environment.womd_control.control import Trajectory

    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    meta = json.loads((args.checkpoint / "manifest.json").read_text())
    if file_hash(args.checkpoint / "replay.npz") != meta["hashes"]["replay.npz"]:
        raise ValueError("source replay checksum mismatch")
    if meta["run_state"]["seed"] != args.seed:
        raise ValueError("warmup must come from the same seed")
    replay = Replay.load(args.checkpoint / "replay.npz", meta["versions"])
    source_report = meta["run_state"]["report"]
    if (
        source_report.get("training_mode") != "pure-sac"
        or source_report.get("bc_checkpoint_sha256") is not None
        or source_report.get("initial_replay_size") != 0
    ):
        raise ValueError("requires original pure SAC self-collection lineage")
    if (
        replay.size < 5000
        or replay.size == replay.capacity
        or replay.cursor != replay.size
        or np.any(replay.arrays["teacher_accepted"][:5000])
    ):
        raise ValueError("requires 5000 self-collected warmup transitions")
    replay.size = replay.cursor = 5000
    current = versions(args.manifest)
    if (
        current["manifest_sha256"] != meta["versions"]["manifest_sha256"]
        or current["contract_sha256"] != meta["versions"]["contract_sha256"]
    ):
        raise ValueError("data/contract mismatch")
    args.output.mkdir(parents=True)
    result = dict(
        status="RUNNING",
        qualified=False,
        scope="PAIRED_FROZEN_WARMUP_SHORT_UPDATES",
        seed=args.seed,
        source_versions=meta["versions"],
        versions=current,
        source_checkpoint_sha256=file_hash(args.checkpoint / "manifest.json"),
        source_replay_sha256=meta["hashes"]["replay.npz"],
        script_sha256=file_hash(__file__),
        updates=args.updates,
        note="First 5000 same-seed self transitions only, identical replay indices and learner RNG in each arm. No new experience, no BC, no teacher. Reset probes are diagnostics, not closed-loop performance or independent test.",
        probes=[],
        arms={},
    )
    bank = []
    visited = set(replay.arrays["scene_id"][:5000])
    for split in ("train", "validation"):
        selected = selected_scenes(args.manifest, split)
        if split == "train":
            selected = [(r, path) for r, path in selected if r["scene_id"] in visited][
                :8
            ]
        for row, path in selected:
            with jax.default_device(cpu):
                env = SACPlanningEnvironment()
                boundary = env.reset(
                    parse_scenario(path.read_bytes()),
                    seed=args.seed,
                    provider_spec=row["provider"],
                )
                obs = tensor_observation(boundary.sac_observation)
                proposal = proposal_from_action(env, boundary.token, np.zeros((30, 2)))
                check = env._check(Trajectory(proposal), env._check_objects())
            result["probes"].append(
                dict(
                    split=split,
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    initial_speed=env.ego.speed,
                    initial_valid=check.valid,
                    initial_reason=check.reason,
                )
            )
            bank.append((env, boundary, jax.device_put(obs, gpu)))
    write_report(args.output / "report.json", result)
    milestones = {0, 1, 10, 100, args.updates}
    if args.updates >= 1000:
        milestones.update((250, 500, 1000))
    for name in args.variants:
        gain, zero, basis, norm = VARIANTS[name]
        replay.rng = np.random.default_rng(args.seed)
        with jax.default_device(gpu):
            state = initialize(
                tensor_observation(CoreObservation.empty()),
                args.seed,
                actor=Actor(mean_gain=gain, mean_basis=basis, head_layer_norm=norm),
                critic=Critic(zero_output=zero),
            )
            forward = jax.jit(state.actor.apply_fn)
            critic1, critic2 = jax.jit(state.q1.apply_fn), jax.jit(state.q2.apply_fn)
        arm = dict(
            status="RUNNING",
            mean_gain=gain,
            zero_q=zero,
            mean_basis=basis,
            actor_head_layer_norm=norm,
            milestones=[],
            updates=0,
        )
        result["arms"][name] = arm
        start = time.monotonic()
        log = (args.output / f"{name}-updates.jsonl").open("w")
        try:
            for count in range(args.updates + 1):
                if count:
                    with jax.default_device(gpu):
                        state, metrics = update(state, replay.sample(256), replay.size)
                    values = {k: float(v) for k, v in metrics.items()}
                    log.write(json.dumps(dict(update=count, **values)) + "\n")
                    log.flush()
                    arm["updates"] = count
                if count not in milestones:
                    continue
                measured = []
                for index, (env, boundary, obs) in enumerate(bank):
                    with jax.default_device(gpu):
                        mu, rho = forward({"params": state.actor.params}, obs)
                        sample = sample_action(
                            mu,
                            rho,
                            observation_speed(obs),
                            jax.random.fold_in(jax.random.PRNGKey(123), index),
                        )[0]
                        minimum_q = float(
                            jax.numpy.minimum(
                                critic1({"params": state.q1.params}, obs, mu),
                                critic2({"params": state.q2.params}, obs, mu),
                            )[0]
                        )
                    record = dict(
                        probe=index,
                        min_q=minimum_q,
                        mean_abs_max=float(abs(mu).max()),
                        actions={},
                    )
                    for action_name, action in (
                        ("mean", np.asarray(mu[0], np.float64)),
                        ("sample", np.asarray(sample[0])),
                    ):
                        with jax.default_device(cpu):
                            proposal = proposal_from_action(env, boundary.token, action)
                            trajectory = Trajectory(proposal)
                            check = env._check(trajectory, env._check_objects())
                            times = np.linspace(0, 3, 301)
                            jerk = np.linalg.norm(
                                trajectory.evaluate(times, 3), axis=1
                            ).max()
                        record["actions"][action_name] = dict(
                            accepted=check.valid,
                            reason=check.reason,
                            violations=list(check.violations),
                            vector_jerk_grid_max=float(jerk),
                        )
                    measured.append(record)
                arm["milestones"].append(dict(update=count, rows=measured))
                arm["elapsed_s"] = time.monotonic() - start
                write_report(args.output / "report.json", result)
                initial = [
                    i for i, r in enumerate(result["probes"]) if r["initial_valid"]
                ]
                print(
                    json.dumps(
                        dict(
                            arm=name,
                            update=count,
                            accepted=sum(
                                r["actions"]["mean"]["accepted"] for r in measured
                            ),
                            retained=sum(
                                measured[i]["actions"]["mean"]["accepted"]
                                for i in initial
                            ),
                            initial_valid=len(initial),
                            max_mean=max(r["mean_abs_max"] for r in measured),
                        )
                    ),
                    flush=True,
                )
            # Diagnostic weights deliberately lack a production checkpoint manifest.
            (args.output / f"{name}-learner.msgpack").write_bytes(
                serialization.to_bytes(state)
            )
            arm["learner_sha256"] = file_hash(args.output / f"{name}-learner.msgpack")
            arm["status"] = "COMPLETE"
        except Exception as exc:
            arm.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            log.close()
            arm["elapsed_s"] = time.monotonic() - start
            write_report(args.output / "report.json", result)
            del state
            jax.clear_caches()
    result["status"] = "COMPLETE"
    if versions(args.manifest) != current:
        raise RuntimeError("runtime changed during diagnostic")
    write_report(args.output / "report.json", result)


if __name__ == "__main__":
    main()
