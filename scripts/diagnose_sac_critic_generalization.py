#!/usr/bin/env python3
"""Terminal-only supervised diagnostic; never a SAC policy or resumable learner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def scene_partition(scene_ids, forced_fit_scenes):
    """Group all repeats of a scene; force the old two action probes into fit."""
    forced = set(forced_fit_scenes)
    return [
        scene not in forced
        and int(hashlib.sha256(("critic_fit_v1|" + scene).encode()).hexdigest(), 16) % 5
        == 0
        for scene in scene_ids
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "diagnostic", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--action-audits", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 100 <= args.steps <= 2000:
        parser.error("diagnostic steps must be 100..2000")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from src.contracts.observation import CoreObservation
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.learner import initialize, gradient_norm
    from src.training.sac.networks import Actor, Critic
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import Replay, unpack_observation
    from src.training.sac.runtime import versions, write_report

    version = versions(args.manifest)
    cp = json.loads((args.checkpoint / "manifest.json").read_text())
    source = json.loads((args.diagnostic / "report.json").read_text())
    assert source["status"] == "COMPLETE" and source["versions"] == version
    assert source["source_checkpoint_sha256"] == file_hash(
        args.checkpoint / "manifest.json"
    )
    assert file_hash(args.checkpoint / "replay.npz") == cp["hashes"]["replay.npz"]
    replay = Replay.load(args.checkpoint / "replay.npz", cp["versions"])
    assert replay.size >= 5000 and replay.cursor == replay.size < replay.capacity
    a = replay.arrays
    assert not a["teacher_accepted"][:5000].any()
    indices = np.flatnonzero(~a["bootstrap"][:5000])
    probes = []
    audit_hashes = {}
    for audit_path in args.action_audits:
        audit = json.loads(audit_path.read_text())
        assert audit["status"] == "COMPLETE" and audit["versions"] == version
        assert audit["replay_sha256"] == cp["hashes"]["replay.npz"]
        model_hash = audit.get("reproduction_model_sha256")
        model = (
            next(m for m in audit["models"] if m["learner_sha256"] == model_hash)
            if model_hash
            else audit["models"][-1]
        )
        rows = {row["index"]: row for row in model["rows"]}
        for case in audit["action_reproductions"]:
            action = case["actions"]["current_mean"]
            assert not action["bootstrap"] and case["index"] in indices
            assert case["scene_id"] == a["scene_id"][case["index"]]
            probes.append(
                dict(
                    index=case["index"],
                    scene_id=case["scene_id"],
                    audit=str(audit_path),
                    raw_u=rows[case["index"]]["current_mean_raw_u"],
                    reward=action["reward"],
                    outcome=action["outcome"],
                    initial_model_q=action["q"],
                )
            )
        audit_hashes[str(audit_path)] = file_hash(audit_path)
    held = np.asarray(
        scene_partition(
            [str(a["scene_id"][i]) for i in indices], [p["scene_id"] for p in probes]
        )
    )
    assert held.any() and (~held).any()
    assert set(a["scene_id"][indices[held]]).isdisjoint(a["scene_id"][indices[~held]])
    arm = source["arms"]["online"]
    if arm.get("critic_action_gain", 1.0) != 1.0:
        raise ValueError("this historical fit protocol does not support input-gain variants")
    weights = args.diagnostic / "online-learner.msgpack"
    digest = file_hash(weights)
    assert digest == arm["learner_sha256"]
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        fresh = initialize(
            tensor_observation(CoreObservation.empty()),
            source["seed"],
            actor=diagnostic_actor(arm, restoring=True),
            critic=Critic(
                zero_output=arm["zero_q"],
                action_coordinates=arm["critic_action_coordinates"],
            ),
        )
        trained = jax.device_put(
            serialization.from_bytes(fresh, weights.read_bytes()), gpu
        )
        obs = jax.device_put(
            unpack_observation(
                a["obs"][indices], a["fields"][indices], a["items"][indices]
            ),
            gpu,
        )
        raw, reward = jax.device_put(a["raw_u"][indices], gpu), jax.device_put(
            a["reward"][indices], gpu
        )
        fit_obs = jax.tree_util.tree_map(lambda x: x[~held], obs)
        probe_indices = np.array([p["index"] for p in probes])
        probe_obs = jax.device_put(
            unpack_observation(
                a["obs"][probe_indices],
                a["fields"][probe_indices],
                a["items"][probe_indices],
            ),
            gpu,
        )
        probe_raw = jax.device_put(
            np.array([p["raw_u"] for p in probes], np.float64), gpu
        )
        probe_rewards = jax.device_put(np.array([p["reward"] for p in probes]), gpu)

    # These are only supervised fits of terminal immediate rewards. No Bellman,
    # actor, alpha, target, replay insertion, or production checkpoint changes.
    @jax.jit
    def fit(q, observation, action, target):
        def loss(params):
            estimate = q.apply_fn({"params": params}, observation, action)
            return jnp.mean((estimate - target) ** 2)

        value, grad = jax.value_and_grad(loss)(q.params)
        updated = q.apply_gradients(grads=grad)
        finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(updated)]
            )
        )
        return updated, value, gradient_norm(grad), finite

    predict = jax.jit(fresh.q1.apply_fn)

    def metrics(q1, q2, observation, action, target):
        prediction = np.minimum(
            np.asarray(predict({"params": q1.params}, observation, action)),
            np.asarray(predict({"params": q2.params}, observation, action)),
        )
        target = np.asarray(target)
        return dict(
            count=len(target),
            mae=float(np.mean(np.abs(prediction - target))),
            signed_error=float(np.mean(prediction - target)),
            q=prediction.tolist(),
        )

    report = dict(
        status="RUNNING",
        qualified=False,
        scope="SUPERVISED_TERMINAL_CRITIC_DIAGNOSTIC_NOT_SAC",
        versions=version,
        script_sha256=file_hash(__file__),
        learner_sha256=digest,
        source_sha256=file_hash(args.diagnostic / "report.json"),
        replay_sha256=cp["hashes"]["replay.npz"],
        action_audit_sha256=audit_hashes,
        steps=args.steps,
        probes=probes,
        fit_indices=indices[~held].tolist(),
        held_indices=indices[held].tolist(),
        fit_scenes=sorted(set(a["scene_id"][indices[~held]])),
        held_scenes=sorted(set(a["scene_id"][indices[held]])),
        fit_rewards=np.asarray(reward[~held]).tolist(),
        held_rewards=np.asarray(reward[held]).tolist(),
        arms={},
        note="Fixed terminal-only supervised diagnostic, not SAC training. Fresh arms have scene-held-out labels; continued arm was previously exposed to all replay states so its holdout is incremental-only. Current-mean actions are genuine prior Waymax terminal branches at fit scenes, not invented labels. No validation scenes or teacher examples. No saved fitted weights are eligible for policy training. Two fixed action-probe scenes were selected on earlier failures and are not an unbiased evaluation.",
    )
    args.output.mkdir(parents=True)
    write_report(args.output / "report.json", report)
    try:
        for name in (
            "continued_records",
            "fresh_records",
            "fresh_zero_action_init",
            "fresh_plus_one_action_per_probe_scene",
        ):
            base = trained if name == "continued_records" else fresh
            q1, q2 = base.q1, base.q2
            if name == "fresh_zero_action_init":

                def reset_action(q):
                    params = jax.tree_util.tree_map(lambda x: x, q.params)
                    params["Dense_0"]["kernel"] = (
                        params["Dense_0"]["kernel"].at[-60:, :].set(0)
                    )
                    return q.replace(params=params)

                q1, q2 = reset_action(q1), reset_action(q2)
            training_obs, training_raw, training_reward = (
                fit_obs,
                raw[~held],
                reward[~held],
            )
            augmented = []
            if name == "fresh_plus_one_action_per_probe_scene":
                seen = set()
                for k, probe in enumerate(probes):
                    if probe["scene_id"] not in seen:
                        seen.add(probe["scene_id"])
                        augmented.append(k)
                training_obs = jax.tree_util.tree_map(
                    lambda x, y: jnp.concatenate([x, y[jnp.asarray(augmented)]]),
                    fit_obs,
                    probe_obs,
                )
                training_raw = jnp.concatenate(
                    [training_raw, probe_raw[jnp.asarray(augmented)]]
                )
                training_reward = jnp.concatenate(
                    [training_reward, probe_rewards[jnp.asarray(augmented)]]
                )
            result = dict(
                status="RUNNING", augmented_probe_indices=augmented, snapshots=[]
            )
            report["arms"][name] = result
            for step in range(args.steps + 1):
                if step in {0, 10, 100, args.steps}:
                    snapshot = dict(
                        step=step,
                        fit=metrics(q1, q2, fit_obs, raw[~held], reward[~held]),
                        scene_holdout=metrics(
                            q1,
                            q2,
                            jax.tree_util.tree_map(lambda x: x[held], obs),
                            raw[held],
                            reward[held],
                        ),
                        action_probes=metrics(
                            q1, q2, probe_obs, probe_raw, probe_rewards
                        ),
                    )
                    result["snapshots"].append(snapshot)
                    write_report(args.output / "report.json", report)
                    print(
                        json.dumps(
                            dict(
                                arm=name,
                                step=step,
                                fit_mae=snapshot["fit"]["mae"],
                                held_mae=snapshot["scene_holdout"]["mae"],
                                probe_mae=snapshot["action_probes"]["mae"],
                            )
                        ),
                        flush=True,
                    )
                if step == args.steps:
                    break
                for first in (True, False):
                    updated, loss, norm, finite = fit(
                        q1 if first else q2, training_obs, training_raw, training_reward
                    )
                    if (
                        not bool(finite)
                        or not np.isfinite([float(loss), float(norm)]).all()
                    ):
                        raise FloatingPointError("nonfinite supervised critic fit")
                    if first:
                        q1 = updated
                    else:
                        q2 = updated
            result["status"] = "COMPLETE"
        assert file_hash(weights) == digest and versions(args.manifest) == version
        for path, expected in audit_hashes.items():
            assert file_hash(path) == expected
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
