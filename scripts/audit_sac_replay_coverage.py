#!/usr/bin/env python3
"""Conditional behavior/current-policy overlap and Q gaps on actual own replay."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sampling_exposure(seed, updates, warmup=5000):
    """Replay.sample's exact index RNG, provided one draw follows each insertion."""
    import numpy as np

    rng = np.random.default_rng(seed)
    counts = np.zeros(warmup + updates, np.int64)
    fresh = 0
    for step in range(1, updates + 1):
        indices = rng.choice(np.arange(warmup + step), 256, replace=True)
        counts += np.bincount(indices, minlength=len(counts))
        fresh += int(np.count_nonzero(indices >= max(warmup, warmup + step - 100)))
    return counts, rng.bit_generator.state, fresh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "warmup-checkpoint", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--reconstructed",
        type=Path,
        help="Recovered online-only replay for a run that did not save its replay",
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import chi2
    from flax import serialization
    from src.contracts.observation import CoreObservation
    from src.training.sac.runtime import versions, write_report
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.replay import Replay, unpack_observation
    from src.training.sac.policy import tensor_observation
    from src.training.sac.networks import (
        Actor,
        Critic,
        normalized_action,
        observation_speed,
    )
    from src.training.sac.learner import initialize
    from scripts.sac_diagnostic_critic import diagnostic_critic, critic_factory_hash

    version = versions(args.manifest)
    source = json.loads((args.diagnostic / "report.json").read_text())
    arm = source["arms"]["online"]
    assert source["status"] == "COMPLETE" and source["versions"] == version
    cp = args.warmup_checkpoint
    meta = json.loads((cp / "manifest.json").read_text())
    assert source["source_checkpoint_sha256"] == file_hash(cp / "manifest.json")
    assert file_hash(cp / "replay.npz") == meta["hashes"]["replay.npz"]
    warmup = Replay.load(cp / "replay.npz", meta["versions"])
    assert warmup.size >= 5000 and warmup.cursor == warmup.size < warmup.capacity
    assert not warmup.arrays["teacher_accepted"][:5000].any()
    if args.reconstructed:
        reconstructed = json.loads((args.reconstructed / "report.json").read_text())
        assert (
            reconstructed["status"] == "COMPLETE"
            and reconstructed["versions"] == version
        )
        assert reconstructed["diagnostic_sha256"] == file_hash(
            args.diagnostic / "report.json"
        )
        replay_path = args.reconstructed / "replay.npz"
        assert file_hash(replay_path) == reconstructed["replay_sha256"]
        online = Replay.load(replay_path, version)
        offset = 0
    else:
        saved = source["saved_replay"]
        replay_path = args.diagnostic / saved["file"]
        assert file_hash(replay_path) == saved["sha256"]
        online = Replay.load(replay_path, version)
        offset = 5000
        for key in warmup.arrays:
            np.testing.assert_array_equal(
                online.arrays[key][:5000], warmup.arrays[key][:5000]
            )
    assert online.size - offset == source["new_transitions"] == source["updates"]
    assert not online.arrays["teacher_accepted"][: online.size].any()
    lineage = None
    if source.get("collection", {}).get("mode") == "actual_failure_branches":
        if args.reconstructed:
            raise ValueError("branch audit requires the original saved replay")
        lineage = [json.loads(line) for line in
                   (args.diagnostic / "ticks.jsonl").read_text().splitlines()]
        if len(lineage) != source["new_transitions"]:
            raise ValueError("incomplete branch collection trace")
        for i, row in enumerate(lineage):
            if row["collection_id"] != i + 1 or row["lineage"]["kind"] not in ("main", "failure_branch"):
                raise ValueError("invalid branch collection lineage")
        np.testing.assert_array_equal(online.arrays["raw_u"][offset:online.size],
                                      np.asarray([r["raw_proposal"]["raw_u"] for r in lineage]))
    counts, rng_state, fresh = sampling_exposure(source["seed"], source["updates"])
    if not args.reconstructed:
        assert online.rng.bit_generator.state == rng_state
    weights = args.diagnostic / "online-learner.msgpack"
    digest = file_hash(weights)
    assert digest == arm["learner_sha256"]
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        state = initialize(
            tensor_observation(CoreObservation.empty()),
            source["seed"],
            actor=Actor(
                mean_gain=arm["mean_gain"],
                mean_basis=arm["mean_basis"],
                head_layer_norm=arm["actor_head_layer_norm"],
                mean_coordinate_scale=arm.get("mean_coordinate_scale", "raw"),
            ),
            critic=diagnostic_critic(arm, restoring=True),
        )
        state = jax.device_put(
            serialization.from_bytes(state, weights.read_bytes()), gpu
        )
        assert int(state.updates) == source["updates"] and int(state.bc_updates) == 0
        forward = jax.jit(state.actor.apply_fn)
        q1, q2 = jax.jit(state.q1.apply_fn), jax.jit(state.q2.apply_fn)
    cutoff = float(np.sqrt(chi2.ppf(0.95, 60) / 60))
    c = 0.02 * np.sqrt(2 * np.pi * np.e)
    rows = []
    for name, arrays, start, end in [
        ("warmup", warmup.arrays, 0, 5000),
        ("online", online.arrays, offset, online.size),
    ]:
        for lo in range(start, end, 128):
            hi = min(lo + 128, end)
            with jax.default_device(gpu):
                obs = jax.device_put(
                    unpack_observation(
                        arrays["obs"][lo:hi],
                        arrays["fields"][lo:hi],
                        arrays["items"][lo:hi],
                    ),
                    gpu,
                )
                raw = jax.device_put(arrays["raw_u"][lo:hi], gpu)
                mu, rho = forward({"params": state.actor.params}, obs)
                speed = observation_speed(obs)
                sigma = 0.005 + 0.025 * jax.nn.sigmoid(rho.astype(jnp.float64))
                mean_initial = normalized_action(mu, speed) * c / 0.02
                residual = normalized_action(raw - mu, speed) * c / sigma[:, None, :]
                norm = np.asarray(jnp.sqrt(jnp.mean(residual**2, axis=(-2, -1))))
                displacement = np.asarray(
                    jnp.sqrt(jnp.mean(mean_initial**2, axis=(-2, -1)))
                )
                qr = np.asarray(
                    jnp.minimum(
                        q1({"params": state.q1.params}, obs, raw),
                        q2({"params": state.q2.params}, obs, raw),
                    )
                )
                qm = np.asarray(
                    jnp.minimum(
                        q1({"params": state.q1.params}, obs, mu),
                        q2({"params": state.q2.params}, obs, mu),
                    )
                )
                sigma = np.asarray(sigma)
                speed = np.asarray(speed)
            for k, index in enumerate(range(lo, hi)):
                global_index = index if name == "warmup" else 5000 + index - offset
                rows.append(
                    dict(
                        group=name,
                        collection_kind=("warmup" if name == "warmup" else
                                         lineage[index-offset]["lineage"]["kind"] if lineage else "main"),
                        index=global_index,
                        scene_id=str(arrays["scene_id"][index]),
                        tick=int(arrays["tick"][index]),
                        speed=float(speed[k]),
                        recorded_action_current_sigma_rms=float(norm[k]),
                        inside_current_95pct=bool(norm[k] <= cutoff),
                        mean_displacement_initial_sigma_rms=float(displacement[k]),
                        sigma_x=float(sigma[k, 0]),
                        sigma_y=float(sigma[k, 1]),
                        q_recorded=float(qr[k]),
                        q_current_mean=float(qm[k]),
                        q_gap=float(qm[k] - qr[k]),
                        reward=float(arrays["reward"][index]),
                        bootstrap=bool(arrays["bootstrap"][index]),
                        training_draws=int(counts[global_index]),
                    )
                )
    summary = {}
    groups = [
        ("warmup", rows[:5000]),
        ("online", rows[5000:]),
        ("online_last100", rows[5000:][-100:]),
    ]
    if lineage:
        for kind in ("main", "failure_branch"):
            selected = [r for r in rows[5000:] if r["collection_kind"] == kind]
            if selected:
                groups.append(("online_" + kind, selected))
    for name, selected in groups:
        terminal = [r for r in selected if not r["bootstrap"] and r["reward"] <= -10]
        summary[name] = dict(
            count=len(selected),
            unique_scenes=len({r["scene_id"] for r in selected}),
            stationary=sum(r["speed"] < 0.1 for r in selected),
            fraction_inside_current_policy_95pct=float(
                np.mean([r["inside_current_95pct"] for r in selected])
            ),
            recorded_action_current_sigma_rms_quantiles=np.quantile(
                [r["recorded_action_current_sigma_rms"] for r in selected],
                [0, 0.5, 0.95, 1],
            ).tolist(),
            mean_displacement_initial_sigma_rms_quantiles=np.quantile(
                [r["mean_displacement_initial_sigma_rms"] for r in selected],
                [0, 0.5, 0.95, 1],
            ).tolist(),
            severe_terminal_count=len(terminal),
            severe_terminal_training_draws=sum(r["training_draws"] for r in terminal),
            training_draws=sum(r["training_draws"] for r in selected),
            never_sampled=sum(r["training_draws"] == 0 for r in selected),
            severe_terminal_recorded_q_mae=(
                float(np.mean([abs(r["q_recorded"] - r["reward"]) for r in terminal]))
                if terminal
                else None
            ),
            severe_terminal_current_mean_q=(
                float(np.mean([r["q_current_mean"] for r in terminal]))
                if terminal
                else None
            ),
        )
    report = dict(
        status="COMPLETE",
        qualified=False,
        scope="CONDITIONAL_REPLAY_POLICY_OVERLAP",
        versions=version,
        diagnostic_sha256=file_hash(args.diagnostic / "report.json"),
        script_sha256=file_hash(__file__),
        critic_factory_sha256=critic_factory_hash(),
        critic_action_coordinates=arm.get("critic_action_coordinates", "raw"),
        collection=source.get("collection"),
        learner_sha256=digest,
        online_replay_sha256=file_hash(replay_path),
        own_warmup_sha256=meta["hashes"]["replay.npz"],
        mean_gain=arm["mean_gain"],
        gaussian_dimensions=60,
        rms_95pct_cutoff=cutoff,
        groups=summary,
        sampling=dict(
            total_draws=int(counts.sum()),
            warmup_draws=int(counts[:5000].sum()),
            online_draws=int(counts[5000:].sum()),
            rolling_latest100_draws=fresh,
            never_sampled_online=int(np.count_nonzero(counts[5000:] == 0)),
            final_replay_online_fraction=source["updates"] / (5000 + source["updates"]),
            saved_rng_verified=not bool(args.reconstructed),
        ),
        note="Overlap means each historical action lies in the FINAL policy 95% Gaussian ellipsoid at that same historical observation. It is not literal support membership (Gaussian support is unbounded), a confidence interval, or a guarantee that fresh actions are covered. Current-mean Q on terminal observations has no known target until that action is actually executed. Sampling counts reconstruct the fixed-seed one-draw-per-insertion protocol; saved final RNG checked where available.",
    )
    assert versions(args.manifest) == version and file_hash(weights) == digest
    args.output.mkdir(parents=True)
    write_report(args.output / "report.json", report)
    with (args.output / "rows.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
