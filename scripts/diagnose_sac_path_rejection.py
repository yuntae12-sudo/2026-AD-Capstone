#!/usr/bin/env python3
"""Read-only final-policy reset probes; no updates or acceptance-rule changes.

Dense-grid derivative maxima are lower bounds, not continuous extrema proofs.
The independent production checker supplies the actual acceptance decision.
"""

import argparse
from collections import Counter
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import numpy as np
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.policy import tensor_observation, proposal_from_action
    from src.training.sac.networks import (
        sample_action,
        observation_speed,
        correlation_matrix,
    )
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.environment.womd_control.control import Trajectory
    from src.scenarios.womd20 import parse_scenario
    from scripts.sac_experiment_runtime import load_model

    version = versions(args.manifest)
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    candidates = {s: selected_scenes(args.manifest, s) for s in ("train", "validation")}
    result = dict(
        scope="FINAL_POLICY_RESET_DIAGNOSTIC",
        qualified=False,
        status="RUNNING",
        versions=version,
        script_sha256=file_hash(__file__),
        note="All validation reset pairs plus first 16 manifest train pairs visited by each seed. GPU inference, CPU production checks. Zero-action is a diagnostic reference, not a replacement policy. Two seeded stochastic draws per state are not an acceptance-rate gate.",
        grid_seconds=0.001,
        rows=[],
        checkpoints={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        previous = json.loads(args.output.read_text())
        for key in ("versions", "script_sha256", "scope"):
            if previous[key] != result[key]:
                raise ValueError(f"probe resume mismatch: {key}")
        result = previous
    completed = {
        (r["seed"], r["split"], r["scene_id"], r["provider"])
        for r in result["rows"]
        if r["action"] == "zero_reference"
    }
    for seed in (0, 1, 2):
        directory = args.run / f"online-seed{seed}"
        report = json.loads((directory / "report.json").read_text())
        if report["status"] != "COMPLETE" or report["versions"] != version:
            raise ValueError("requires completed matching-runtime training")
        visited = {(r["scene_id"], r["provider"]) for r in report["episodes"]}
        checkpoint = directory / report["latest_checkpoint"]
        previous_checkpoint = result["checkpoints"].get(str(seed))
        if previous_checkpoint and previous_checkpoint["sha256"] != file_hash(
            checkpoint / "manifest.json"
        ):
            raise ValueError("checkpoint changed during diagnostic resume")
        with jax.default_device(gpu):
            state, _ = load_model(checkpoint, version, seed)
            actor = jax.jit(state.actor.apply_fn)
            q1, q2 = jax.jit(state.q1.apply_fn), jax.jit(state.q2.apply_fn)
        result["checkpoints"][str(seed)] = dict(
            path=str(checkpoint.resolve()),
            sha256=file_hash(checkpoint / "manifest.json"),
            updates=int(state.updates),
            bc_updates=int(state.bc_updates),
        )
        for split, selected in candidates.items():
            if split == "train":
                selected = [
                    pair
                    for pair in selected
                    if (pair[0]["scene_id"], pair[0]["provider"]) in visited
                ][:16]
            for index, (row, path) in enumerate(selected):
                if (seed, split, row["scene_id"], row["provider"]) in completed:
                    continue
                with jax.default_device(cpu):
                    env = SACPlanningEnvironment()
                    boundary = env.reset(
                        parse_scenario(path.read_bytes()),
                        seed=seed,
                        provider_spec=row["provider"],
                    )
                    obs = tensor_observation(boundary.sac_observation)
                with jax.default_device(gpu):
                    obs = jax.device_put(obs, gpu)
                    mu, rho = actor({"params": state.actor.params}, obs)
                    key = jax.random.fold_in(
                        jax.random.PRNGKey(seed),
                        index + (1000 if split == "validation" else 0),
                    )
                    draws = [
                        np.asarray(
                            sample_action(mu, rho, observation_speed(obs), k)[0][0]
                        )
                        for k in jax.random.split(key, 2)
                    ]
                    sigma = 0.005 + 0.025 * jax.nn.sigmoid(rho)
                    g = jax.numpy.clip((observation_speed(obs) / 2) ** 3, 0.002, 1)
                    sd = (
                        g[:, None, None]
                        * jax.numpy.sqrt(
                            jax.numpy.sum(correlation_matrix() ** 2, axis=1)
                        )[None, :, None]
                        * sigma[:, None, :]
                    )
                    actions = [
                        ("mean", np.asarray(mu[0], np.float64)),
                        ("sample0", draws[0]),
                        ("sample1", draws[1]),
                        ("zero_reference", np.zeros((30, 2))),
                    ]
                    qvalues = {}
                    for name, action in actions:
                        a = jax.device_put(action[None], gpu)
                        qvalues[name] = float(
                            jax.numpy.minimum(
                                q1({"params": state.q1.params}, obs, a),
                                q2({"params": state.q2.params}, obs, a),
                            )[0]
                        )
                for name, action in actions:
                    with jax.default_device(cpu):
                        proposal = proposal_from_action(env, boundary.token, action)
                        trajectory = Trajectory(proposal)
                        checked = env._check(trajectory, env._check_objects())
                        times = np.linspace(0, 3, 3001)
                        speed, accel, jerk = [
                            np.linalg.norm(trajectory.evaluate(times, d), axis=1)
                            for d in (1, 2, 3)
                        ]
                    result["rows"].append(
                        dict(
                            seed=seed,
                            split=split,
                            scene_id=row["scene_id"],
                            provider=row["provider"],
                            action=name,
                            initial_speed=env.ego.speed,
                            accepted=checked.valid,
                            reason=checked.reason,
                            violations=list(checked.violations),
                            raw_abs_max=float(np.max(np.abs(action))),
                            noise_std_max=float(sd.max()),
                            predicted_min_q=qvalues[name],
                            max_distance_m=float(
                                np.linalg.norm(
                                    proposal.xy_world - proposal.anchor_world_xy, axis=1
                                ).max()
                            ),
                            grid_speed_max_mps=float(speed.max()),
                            grid_acceleration_max_mps2=float(accel.max()),
                            grid_vector_jerk_max_mps3=float(jerk.max()),
                        )
                    )
                write_report(args.output, result)
            print(
                json.dumps(dict(seed=seed, split=split, pairs=len(selected))),
                flush=True,
            )
        del state
        jax.clear_caches()
    if versions(args.manifest) != version:
        raise RuntimeError("runtime changed during probe")
    result["status"] = "COMPLETE"
    result["reason_counts"] = dict(
        Counter(r["reason"] for r in result["rows"] if r["action"] == "mean")
    )
    write_report(args.output, result)


if __name__ == "__main__":
    main()
