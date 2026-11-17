#!/usr/bin/env python3
"""SAC experiment with explicit pure-SAC or auxiliary BC-SAC initialization.

CPU Waymax and GPU learner coexist with explicit device contexts. Validation
never enters replay. Recovery saves only at episode boundaries, every ~1000
transitions; experiment checkpoints every ~10000, evaluation exactly every5000.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "synthetic", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("pure-sac", "bc-sac"), default="pure-sac")
    for name in ("bc", "gate"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pure = args.mode == "pure-sac"
    if pure and (args.bc is not None or args.gate is not None):
        parser.error("pure SAC forbids BC checkpoint and BC gate inputs")
    if not pure and (args.bc is None or args.gate is None):
        parser.error("auxiliary BC-SAC requires --bc and --gate")
    minimum = 15000 if pure else 10000
    if args.transitions is None:
        args.transitions = minimum
    if args.transitions < minimum or args.transitions % 5000:
        parser.error(f"experiment budget must be >={minimum} and a multiple of5000")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import numpy as np
    from flax import serialization
    from src.contracts.planning import to_wire
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.training.sac.runtime import versions, write_report, selected_scenes
    from src.training.sac.checkpoint import save_checkpoint, load_checkpoint, file_hash
    from src.training.sac.collector import collect_step
    from src.training.sac.learner import update
    from scripts.pure_sac_protocol import (
        WARMUP,
        expected_updates,
        protocol_metadata,
        require_synthetic_report,
        validate_pure_state,
        pure_update,
    )
    from scripts.sac_experiment_runtime import (
        require_bc_checkpoint,
        load_model,
        model_proposer,
        evaluate,
        runner_hash,
        restore_trace,
        checkpoint_metadata,
    )

    if pure:
        version = versions(args.manifest)
        require_synthetic_report(
            json.loads(args.synthetic.read_text()),
            version,
            file_hash(Path(__file__).with_name("validate_sac_exploration.py")),
        )
    else:
        version, _ = require_bc_checkpoint(
            args.bc, args.gate, args.synthetic, args.manifest, args.seed
        )
    runner = runner_hash()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpu = jax.devices("cpu")[0]
    learner_device = jax.devices()[0]
    if learner_device.platform != "gpu":
        raise ValueError("online experiment requires GPU learner and CPU simulator")
    if not pure:
        gate_devices = json.loads(args.gate.read_text()).get("devices", {})
        if gate_devices != dict(simulator=str(cpu), inference=str(learner_device)):
            raise ValueError("BC gate and online execution must use the same devices")
    trace = args.output / "ticks.jsonl"
    metrics_trace = args.output / "updates.jsonl"
    candidates = selected_scenes(args.manifest, "train")
    rng = np.random.default_rng(args.seed)
    order, cursor, cycles = None, 0, 0
    with jax.default_device(learner_device):
        if pure:
            from src.contracts.observation import CoreObservation
            from src.training.sac.policy import tensor_observation
            from src.training.sac.learner import initialize
            from src.training.sac.replay import Replay

            state = initialize(tensor_observation(CoreObservation.empty()), args.seed)
            replay = Replay(seed=args.seed, versions=version)
            validate_pure_state(state, replay, 0)
        else:
            state, _ = load_model(args.bc, version, args.seed)
            state, replay, _ = load_checkpoint(args.bc, state, version)
        state = jax.device_put(state, learner_device)
    if not pure and (
        int(state.updates) != 0 or int(state.bc_updates) < 5000 or replay.size < 20000
    ):
        raise ValueError("BC checkpoint counters/replay invalid")
    policy, proposer = model_proposer(state.actor, args.seed, False, learner_device)
    report = dict(
        status="RUNNING",
        phase="ONLINE_SAC",
        training_mode=args.mode,
        protocol=protocol_metadata() if pure else None,
        initial_replay_size=replay.size,
        warmup_transitions=WARMUP if pure else 0,
        seed=args.seed,
        versions=version,
        runner_sha256=runner,
        budget=args.transitions,
        valid_online_transitions=0,
        updates=0,
        alpha_bound_updates=0,
        episodes=[],
        evaluations=[],
        bc_checkpoint_sha256=file_hash(args.bc / "manifest.json") if args.bc else None,
        gate_sha256=file_hash(args.gate) if args.gate else None,
        synthetic_sha256=file_hash(args.synthetic),
        elapsed_s=0.0,
        group_status="scenario_fallback_overlap_unverified",
        devices=dict(simulator=str(cpu), learner=str(learner_device)),
    )
    next_eval, next_checkpoint, next_recovery = 5000, 10000, 1000
    if args.resume:
        pointer = json.loads((args.output / "latest_recovery.json").read_text())
        checkpoint = (args.output / pointer["path"]).resolve()
        if not checkpoint.is_relative_to(args.output.resolve()):
            raise ValueError("invalid recovery path")
        checkpoint_metadata(checkpoint, version)
        # Free the initial replay before restoring the populated online replay.
        del replay
        with jax.default_device(learner_device):
            state, replay, run = load_checkpoint(checkpoint, state, version)
            state = jax.device_put(state, learner_device)
        old_report = run["report"]
        for key in (
            "runner_sha256",
            "budget",
            "seed",
            "bc_checkpoint_sha256",
            "gate_sha256",
            "training_mode",
            "protocol",
            "initial_replay_size",
            "warmup_transitions",
            "synthetic_sha256",
        ):
            if old_report[key] != report[key]:
                raise ValueError(f"online resume mismatch: {key}")
        if run["phase"] != "ONLINE_SAC" or int(state.updates) != (
            expected_updates(old_report["valid_online_transitions"])
            if pure
            else old_report["valid_online_transitions"]
        ):
            raise ValueError("online recovery update/transition mismatch")
        report = old_report
        if pure:
            validate_pure_state(state, replay, report["valid_online_transitions"])
        rng.bit_generator.state = run["scene_rng"]
        order, cursor, cycles = run["order"], run["cursor"], run["cycles"]
        next_eval, next_checkpoint, next_recovery = (
            run["next_eval"],
            run["next_checkpoint"],
            run["next_recovery"],
        )
        policy.rng = jax.device_put(
            np.asarray(run["policy_rng"], np.uint32), learner_device
        )
        restore_trace(trace, run["trace_bytes"])
        restore_trace(metrics_trace, run["metrics_bytes"])
        report["status"] = "RUNNING"
    started = time.monotonic()
    elapsed_before = report["elapsed_s"]

    def save_boundary(path):
        if versions(args.manifest) != version or runner_hash() != runner:
            raise RuntimeError("runtime changed during online experiment")
        if pure:
            validate_pure_state(state, replay, report["valid_online_transitions"])
        report["elapsed_s"] = elapsed_before + time.monotonic() - started
        report["updates"] = int(state.updates)
        run = dict(
            phase="ONLINE_SAC",
            seed=args.seed,
            report=report,
            scene_rng=rng.bit_generator.state,
            policy_rng=np.asarray(policy.rng).tolist(),
            order=order,
            cursor=cursor,
            cycles=cycles,
            next_eval=next_eval,
            next_checkpoint=next_checkpoint,
            next_recovery=next_recovery,
            trace_bytes=trace.stat().st_size if trace.exists() else 0,
            metrics_bytes=metrics_trace.stat().st_size if metrics_trace.exists() else 0,
        )
        save_checkpoint(path, state, replay, version, run, episode_boundary=True)
        write_report(
            args.output / "latest_recovery.json",
            dict(path=str(path.relative_to(args.output))),
        )

    try:
        if not args.resume:
            save_boundary(args.output / "checkpoints" / "step-00000000")
        write_report(args.output / "report.json", report)
        while report["valid_online_transitions"] < args.transitions:
            if order is None:
                order = rng.permutation(len(candidates)).tolist()
                cursor = 0
            from_cycle_start = cursor == 0
            cycle_start = report["valid_online_transitions"]
            for index in range(cursor, len(order)):
                row, path = candidates[order[index]]
                with jax.default_device(cpu):
                    e = SACPlanningEnvironment()
                    b = e.reset(
                        parse_scenario(path.read_bytes()),
                        seed=args.seed,
                        provider_spec=row["provider"],
                    )
                    if e._state.sim_trajectory.x.device != cpu:
                        raise AssertionError("simulator left its CPU device")
                episode = dict(
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    ticks=0,
                    valid_transitions=0,
                    accepted=0,
                    stationary_ticks=0,
                    reasons={},
                    sources={},
                    return_training=0.0,
                    outcome=None,
                )
                report["episodes"].append(episode)
                while True:
                    policy.actor = state.actor
                    with jax.default_device(cpu):
                        step = collect_step(e, b, proposer, replay)
                    r, b = step.result, step.boundary
                    episode["ticks"] += 1
                    episode["accepted"] += int(r.check_result.valid)
                    episode["stationary_ticks"] += int(
                        r.sample_valid and e.ego.speed < 0.1
                    )
                    for key, value in (
                        ("reasons", r.check_result.reason),
                        ("sources", r.source.value),
                    ):
                        episode[key][value] = episode[key].get(value, 0) + 1
                    episode["return_training"] += step.reward.training
                    with trace.open("a") as stream:
                        stream.write(
                            json.dumps(
                                dict(
                                    episode_uid=e.episode_uid,
                                    scene_id=row["scene_id"],
                                    provider=row["provider"],
                                    tick=e.tick,
                                    raw_proposal=to_wire(r.raw_proposal),
                                    source=r.source.value,
                                    check_valid=r.check_result.valid,
                                    check_reason=r.check_result.reason,
                                    actual_control=to_wire(r.actual_control),
                                    event_flags=r.event_flags,
                                    sample_valid=r.sample_valid,
                                    reward=step.reward.training,
                                    next_token=to_wire(b.token),
                                    terminated=r.terminated,
                                    truncated=r.truncated,
                                    terminal_reason=(
                                        r.terminal_reason.value
                                        if r.terminal_reason
                                        else None
                                    ),
                                ),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    if step.transition is not None:
                        report["valid_online_transitions"] += 1
                        episode["valid_transitions"] += 1
                        with jax.default_device(learner_device):
                            if pure:
                                state, metrics = pure_update(
                                    state, replay, report["valid_online_transitions"]
                                )
                            else:
                                state, metrics = update(
                                    state, replay.sample(256), replay.size
                                )
                        if metrics is not None:
                            report["updates"] = int(state.updates)
                            report["alpha_bound_updates"] += int(
                                metrics["alpha_at_bound"]
                            )
                            with metrics_trace.open("a") as stream:
                                stream.write(
                                    json.dumps(
                                        dict(
                                            step=report["valid_online_transitions"],
                                            **{k: float(v) for k, v in metrics.items()},
                                        ),
                                        allow_nan=False,
                                    )
                                    + "\n"
                                )
                        if report["valid_online_transitions"] == next_eval:
                            _, deterministic = model_proposer(
                                state.actor, args.seed, True, learner_device
                            )
                            import hashlib

                            policy_sha = hashlib.sha256(
                                serialization.to_bytes(state.actor.params)
                            ).hexdigest()
                            destination = (
                                args.output
                                / "evaluations"
                                / f"step-{next_eval:08d}-{time.time_ns()}.json"
                            )
                            result = evaluate(
                                args.manifest,
                                destination,
                                f"{'PURE_SAC' if pure else 'SAC'}-{next_eval}",
                                deterministic,
                                args.seed,
                                cpu,
                                policy_sha,
                                expected_version=version,
                            )
                            if not result.get("qualified"):
                                raise RuntimeError("periodic evaluation failed")
                            report["evaluations"].append(
                                dict(
                                    step=next_eval,
                                    report=str(destination.relative_to(args.output)),
                                    policy_sha256=policy_sha,
                                )
                            )
                            next_eval += 5000
                    if r.terminated or r.truncated:
                        episode["outcome"] = r.terminal_reason.value
                        break
                cursor = index + 1
                count = report["valid_online_transitions"]
                report.update(
                    updates=int(state.updates),
                    elapsed_s=elapsed_before + time.monotonic() - started,
                    alpha_bound_fraction=report["alpha_bound_updates"]
                    / max(int(state.updates), 1),
                    peak_process_rss_bytes=resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * 1024,
                )
                if int(state.updates) != (expected_updates(count) if pure else count):
                    raise AssertionError("one-update-per-transition contract violated")
                if count >= next_checkpoint or count >= args.transitions:
                    while next_checkpoint <= count:
                        next_checkpoint += 10000
                    next_recovery = count + 1000
                    destination = args.output / "checkpoints" / f"step-{count:08d}"
                    report["latest_checkpoint"] = str(
                        destination.relative_to(args.output)
                    )
                    save_boundary(destination)
                elif count >= next_recovery:
                    next_recovery = count + 1000
                    destination = args.output / "recovery" / f"step-{count:08d}"
                    save_boundary(destination)
                    for old in sorted(destination.parent.glob("step-*"))[:-2]:
                        shutil.rmtree(old)
                write_report(args.output / "report.json", report)
                print(
                    json.dumps(
                        dict(
                            episode=episode,
                            transitions=count,
                            updates=int(state.updates),
                        )
                    ),
                    flush=True,
                )
                if count >= args.transitions:
                    break
            if report["valid_online_transitions"] == cycle_start and from_cycle_start:
                raise RuntimeError("no valid online transitions in train pass")
            order, cursor = None, 0
            cycles += 1
        report.update(status="COMPLETE", qualified=True)
    except BaseException as exc:
        report.update(
            status="ERROR",
            qualified=False,
            error=f"{type(exc).__name__}: {exc}",
            updates=int(state.updates),
            pending_update=(
                int(state.updates)
                < (
                    expected_updates(report["valid_online_transitions"])
                    if pure
                    else report["valid_online_transitions"]
                )
            ),
        )
        raise
    finally:
        report["elapsed_s"] = elapsed_before + time.monotonic() - started
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
