#!/usr/bin/env python3
"""BC mean rollouts, then 256-sample probes at predeclared validation ticks.

Validation uses every manifest scene/provider, never test. Stochastic probes at
ticks 0/10/30 are an initial connection gate, not coverage of every rollout state.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROBE_TICKS = (0, 10, 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output required; partial gate evidence is preserved")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from flax import serialization
    import jax
    import numpy as np
    from src.contracts.observation import CoreObservation
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.collector import collect_step
    from src.training.sac.gates import probe_boundary, require_training_gate
    from src.training.sac.learner import initialize
    from src.training.sac.policy import Policy, tensor_observation
    from src.training.sac.replay import pack_observation
    from src.training.sac.runtime import selected_scenes, versions, write_report

    cpu_device = jax.devices("cpu")[0]
    inference_device = jax.devices()[0]
    version = versions(args.manifest)
    meta = json.loads((args.checkpoint / "manifest.json").read_text())
    run = meta["run_state"]
    if meta["versions"] != version or not meta["episode_boundary"]:
        raise ValueError("incompatible checkpoint")
    if set(meta["hashes"]) != {"learner.msgpack", "replay.npz"}:
        raise ValueError("incomplete checkpoint manifest")
    for name, sha in meta["hashes"].items():
        if file_hash(args.checkpoint / name) != sha:
            raise ValueError("checkpoint checksum mismatch")
    if run["phase"] != "BC" or (
        not args.smoke and (not run["qualified"] or run["bc_updates"] < 5000)
    ):
        raise ValueError("requires completed 5000-minibatch BC")
    state = serialization.from_bytes(
        initialize(tensor_observation(CoreObservation.empty()), run["seed"]),
        (args.checkpoint / "learner.msgpack").read_bytes(),
    )
    state = jax.device_put(state, inference_device)
    if int(state.bc_updates) != run["bc_updates"] or int(state.updates) != 0:
        raise ValueError("BC learner counters mismatch")
    synthetic = json.loads(args.synthetic.read_text())
    if synthetic["versions"] != version or synthetic["scope"] != "SYNTHETIC_KINEMATICS":
        raise ValueError("synthetic gate provenance mismatch")
    report = dict(
        status="RUNNING_MEAN",
        scope="SMOKE" if args.smoke else "BC_VALIDATION",
        versions=version,
        checkpoint_sha256=file_hash(args.checkpoint / "manifest.json"),
        runner_sha256=file_hash(__file__),
        seed=run["seed"],
        devices=dict(simulator=str(cpu_device), inference=str(inference_device)),
        synthetic_sha256=file_hash(args.synthetic),
        synthetic_cases=synthetic["cases"],
        probe_ticks=PROBE_TICKS,
        episodes=[],
        states=[],
        passed=False,
        group_status="scenario_fallback_overlap_unverified",
    )
    with jax.default_device(inference_device):
        policy = Policy(state.actor, seed=run["seed"], deterministic=True)

    def propose(e, token):
        with jax.default_device(inference_device):
            return policy.propose(e, token)

    def reset(row, path):
        with jax.default_device(cpu_device):
            e = SACPlanningEnvironment()
            b = e.reset(
                parse_scenario(path.read_bytes()),
                seed=run["seed"],
                provider_spec=row["provider"],
            )
        if e._state.sim_trajectory.x.device != cpu_device:
            raise AssertionError("gate simulator left CPU")
        return e, b

    def collect(e, boundary):
        with jax.default_device(cpu_device):
            return collect_step(e, boundary, propose)

    candidates = selected_scenes(args.manifest, "validation")
    # Smoke keeps both behaviors but never qualifies a gate.
    if args.smoke:
        candidates = [
            next(c for c in candidates if c[0]["provider"] == p)
            for p in ("fixed_keep", "fixed_follow")
        ]

    def fingerprint(observation):
        digest = hashlib.sha256()
        for a in pack_observation(observation):
            digest.update(a.tobytes())
        return digest.hexdigest()

    def mean_summary():
        result = {}
        for behavior in ("KEEP", "FOLLOW"):
            episodes = [e for e in report["episodes"] if e["behavior"] == behavior]
            ticks = sum(e["ticks"] for e in episodes)
            accepted = sum(e["accepted"] for e in episodes)
            result[behavior] = dict(
                episodes=len(episodes),
                ticks=ticks,
                accepted=accepted,
                pass_rate=accepted / ticks if ticks else 0.0,
                reasons=dict(sum((Counter(e["reasons"]) for e in episodes), Counter())),
            )
        return result

    write_report(args.output, report)
    try:
        for row, path in candidates:
            e, boundary = reset(row, path)
            episode = dict(
                scene_id=row["scene_id"],
                provider=row["provider"],
                behavior=e.command.draft.maneuver.value,
                ticks=0,
                accepted=0,
                return_training=0.0,
                stationary_ticks=0,
                distance_traveled_m=0.0,
                reasons={},
                sources={},
                events={},
                probe_observations={},
                outcome=None,
            )
            report["episodes"].append(episode)
            previous_position = e.ego.position.copy()
            while True:
                if e.tick in PROBE_TICKS:
                    episode["probe_observations"][str(e.tick)] = fingerprint(
                        boundary.sac_observation
                    )
                collected = collect(e, boundary)
                result, boundary = collected.result, collected.boundary
                episode["ticks"] += 1
                episode["accepted"] += int(result.check_result.valid)
                episode["return_training"] += collected.reward.training
                for field, values in (
                    ("reasons", [result.check_result.reason]),
                    ("sources", [result.source.value]),
                    ("events", result.event_flags),
                ):
                    for value in values:
                        episode[field][value] = episode[field].get(value, 0) + 1
                if result.sample_valid:
                    episode["stationary_ticks"] += int(e.ego.speed < 0.1)
                    episode["distance_traveled_m"] += float(
                        np.linalg.norm(e.ego.position - previous_position)
                    )
                    previous_position = e.ego.position.copy()
                if result.terminated or result.truncated:
                    episode["outcome"] = result.terminal_reason.value
                    break
                if args.smoke and episode["ticks"] >= 2:
                    episode["outcome"] = "SMOKE_BATCH_BOUNDARY"
                    break
            episode["unreached_probe_ticks"] = [
                t for t in PROBE_TICKS if str(t) not in episode["probe_observations"]
            ]
            report["deterministic_mean"] = mean_summary()
            write_report(args.output, report)
            print(json.dumps(episode), flush=True)
        if any(e["outcome"] == "DATA_CORRUPTION" for e in report["episodes"]):
            report.update(status="FAILED_DATA_INTEGRITY")
            return 2
        if args.smoke or any(r["pass_rate"] < 0.9 for r in mean_summary().values()):
            report.update(status="SMOKE_COMPLETE" if args.smoke else "FAILED_MEAN")
            return 0 if args.smoke else 2
        if not synthetic.get("qualified") or not synthetic.get("passed"):
            report.update(status="FAILED_SYNTHETIC")
            return 2
        report["status"] = "RUNNING_STOCHASTIC"
        # Repeat deterministic trajectories; fingerprint equality prevents probing
        # a different state due to non-reproducible resets or policy execution.
        for index, ((row, path), episode) in enumerate(
            zip(candidates, report["episodes"])
        ):
            e, boundary = reset(row, path)
            while e.tick <= max(PROBE_TICKS):
                if e.tick in PROBE_TICKS:
                    if fingerprint(boundary.sac_observation) != episode[
                        "probe_observations"
                    ].get(str(e.tick)):
                        raise ValueError(
                            "validation replay state differs from mean rollout"
                        )
                    with jax.default_device(inference_device):
                        key = jax.random.fold_in(
                            jax.random.PRNGKey(run["seed"]), index * 1000 + e.tick
                        )
                        report["states"].append(
                            probe_boundary(e, boundary, state.actor, key)
                        )
                    write_report(args.output, report)
                if e.tick == max(PROBE_TICKS):
                    break
                collected = collect(e, boundary)
                boundary = collected.boundary
                if collected.result.terminated or collected.result.truncated:
                    break
        report["scope"] = "QUALIFIED_BC_GATE"
        try:
            require_training_gate(report, version, report["checkpoint_sha256"])
        except ValueError as exc:
            report.update(status="FAILED_STOCHASTIC", error=str(exc))
            return 2
        report.update(status="PASSED", passed=True)
        return 0
    except BaseException as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if versions(args.manifest) != version:
            report.update(
                status="ERROR", passed=False, error="runtime changed during gate"
            )
        write_report(args.output, report)


if __name__ == "__main__":
    raise SystemExit(main())
