#!/usr/bin/env python3
"""Exact terminal-target errors on the original 5,000 self-warmup transitions."""

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
    for name in ("checkpoint", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--diagnostics", type=Path, nargs="+", required=True)
    p.add_argument("--reproduce-count", type=int, default=2)
    p.add_argument(
        "--reference-cases",
        type=Path,
        help="Use the same train terminal indices as an earlier audit",
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.reproduce_count < 0:
        p.error("nonnegative reproduction count required")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import numpy as np
    from flax import serialization
    from src.contracts.observation import CoreObservation
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.learner import initialize
    from src.training.sac.networks import Actor, Critic
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import Replay, unpack_observation
    from src.training.sac.runtime import versions, write_report

    version = versions(args.manifest)
    meta = json.loads((args.checkpoint / "manifest.json").read_text())
    origin = meta["run_state"]["report"]
    assert (
        origin["training_mode"] == "pure-sac" and origin["warmup_transitions"] == 5000
    )
    assert origin["initial_replay_size"] == 0 and origin["bc_checkpoint_sha256"] is None
    assert file_hash(args.checkpoint / "replay.npz") == meta["hashes"]["replay.npz"]
    for key in ("manifest_sha256", "contract_sha256"):
        assert meta["versions"][key] == version[key]
    replay = Replay.load(args.checkpoint / "replay.npz", meta["versions"])
    assert 5000 <= replay.size < replay.capacity and replay.cursor == replay.size
    assert not replay.arrays["teacher_accepted"][:5000].any()
    indices = np.flatnonzero(~replay.arrays["bootstrap"][:5000])
    a = replay.arrays
    rewards = a["reward"][indices]
    report = dict(
        status="RUNNING",
        qualified=False,
        scope="READ_ONLY_SELF_WARMUP_TERMINAL_Q_ERRORS",
        versions=version,
        source_versions=meta["versions"],
        replay_sha256=meta["hashes"]["replay.npz"],
        script_sha256=file_hash(__file__),
        prefix_size=5000,
        terminal_transitions=len(indices),
        models=[],
        note="Every original self-warmup terminal transition, evaluated at its recorded raw action. Target equals immediate reward exactly because bootstrap is false. This is training-data fit, not generalization; new online 1,000 transitions are not included. Reward <= -10 is a severe-penalty group, not a reconstructed collision event label.",
    )
    gpu = jax.devices("gpu")[0]
    for diagnostic in args.diagnostics:
        source = json.loads((diagnostic / "report.json").read_text())
        arm = source["arms"]["online"]
        assert source["status"] == "COMPLETE" and source["versions"] == version
        assert source["source_checkpoint_sha256"] == file_hash(
            args.checkpoint / "manifest.json"
        )
        weights = diagnostic / "online-learner.msgpack"
        digest = file_hash(weights)
        assert digest == arm["learner_sha256"]
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
                critic=Critic(
                    zero_output=arm["zero_q"],
                    action_coordinates=arm.get("critic_action_coordinates", "raw"),
                ),
            )
            state = jax.device_put(
                serialization.from_bytes(state, weights.read_bytes()), gpu
            )
            assert int(state.bc_updates) == 0 and int(state.updates) == arm["updates"]
            q1, q2 = jax.jit(state.q1.apply_fn), jax.jit(state.q2.apply_fn)
            obs = unpack_observation(
                a["obs"][indices], a["fields"][indices], a["items"][indices]
            )
            obs = jax.device_put(obs, gpu)
            u = jax.device_put(a["raw_u"][indices], gpu)
            qa = np.asarray(q1({"params": state.q1.params}, obs, u))
            qb = np.asarray(q2({"params": state.q2.params}, obs, u))
            mu, _ = state.actor.apply_fn({"params": state.actor.params}, obs)
            mean_q = np.asarray(
                jax.numpy.minimum(
                    q1({"params": state.q1.params}, obs, mu),
                    q2({"params": state.q2.params}, obs, mu),
                )
            )
            mean_actions = np.asarray(mu, dtype=np.float64)
        q = np.minimum(qa, qb)
        assert np.isfinite(q).all() and file_hash(weights) == digest
        groups = {}
        for name, mask in [
            ("all", np.ones(len(q), bool)),
            ("severe_penalty", rewards <= -10),
            ("other", rewards > -10),
        ]:
            error = q[mask] - rewards[mask]
            groups[name] = dict(
                count=int(mask.sum()),
                mean_abs_error=float(np.abs(error).mean()) if len(error) else None,
                mean_error=float(error.mean()) if len(error) else None,
                mean_q=float(q[mask].mean()) if len(error) else None,
                mean_reward=float(rewards[mask].mean()) if len(error) else None,
                current_mean_q=float(mean_q[mask].mean()) if len(error) else None,
            )
        report["models"].append(
            dict(
                diagnostic=str(diagnostic),
                learner_sha256=digest,
                updates=int(state.updates),
                mean_gain=arm["mean_gain"],
                groups=groups,
                rows=[
                    dict(
                        index=int(ix),
                        scene_id=str(a["scene_id"][ix]),
                        tick=int(a["tick"][ix]),
                        reward=float(rewards[k]),
                        q1=float(qa[k]),
                        q2=float(qb[k]),
                        min_q=float(q[k]),
                        current_mean_q=float(mean_q[k]),
                        current_mean_raw_u=mean_actions[k].tolist(),
                        terminated=bool(a["terminated"][ix]),
                        truncated=bool(a["truncated"][ix]),
                    )
                    for k, ix in enumerate(indices)
                ],
            )
        )
        write_report(args.output, report)
    # Deliberately selected counterexamples, not an unbiased training evaluation.
    from src.training.sac.runtime import selected_scenes
    from src.training.sac.replay import pack_observation
    from src.training.sac.collector import collect_step
    from src.training.sac.policy import proposal_from_action
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.contracts.planning import Maneuver
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint

    selected = {
        (r["scene_id"], r["provider"]): path
        for r, path in selected_scenes(args.manifest, "train")
    }
    # Last explicitly supplied model is the candidate, including equal-budget pairs.
    latest = report["models"][-1]
    report["reproduction_model_sha256"] = latest["learner_sha256"]
    candidates = sorted(
        (r for r in latest["rows"] if r["reward"] <= -10),
        key=lambda r: r["current_mean_q"] - r["min_q"],
        reverse=True,
    )
    selection = "largest current-mean minus stored-action Q gaps, distinct scenes"
    if args.reference_cases:
        reference = json.loads(args.reference_cases.read_text())
        assert (
            reference["status"] == "COMPLETE"
            and reference["replay_sha256"] == meta["hashes"]["replay.npz"]
        )
        assert reference["versions"] == version
        by_index = {row["index"]: row for row in latest["rows"]}
        candidates = [
            by_index[row["index"]] for row in reference["action_reproductions"]
        ]
        if len(candidates) < args.reproduce_count:
            raise ValueError("not enough reference cases")
        report["reference_cases_sha256"] = file_hash(args.reference_cases)
        selection = "fixed terminal indices from earlier train audit, not reselected for this candidate"
    report["action_reproductions"] = []
    seen = set()
    for row in candidates:
        if len(report["action_reproductions"]) >= args.reproduce_count:
            break
        if row["scene_id"] in seen:
            continue
        seen.add(row["scene_id"])
        ix = row["index"]
        episode_indices = np.flatnonzero(
            a["episode_uid"][:5000] == a["episode_uid"][ix]
        )
        assert a["tick"][episode_indices[0]] == 0 and episode_indices[-1] == ix
        core = unpack_observation(a["obs"][ix], a["fields"][ix], a["items"][ix])
        maneuver = list(Maneuver)[
            int(np.argmax(core["features"]["command"][:5]))
        ].value.lower()
        provider = "fixed_" + maneuver
        with jax.default_device(jax.devices("cpu")[0]):
            e = SACPlanningEnvironment()
            boundary = e.reset(
                parse_scenario(selected[(row["scene_id"], provider)].read_bytes()),
                seed=meta["run_state"]["seed"],
                provider_spec=provider,
            )
            for index in episode_indices:
                packed = pack_observation(boundary.sac_observation)
                np.testing.assert_allclose(
                    packed[0], a["obs"][index], atol=1e-6, rtol=0
                )
                np.testing.assert_array_equal(packed[1], a["fields"][index])
                np.testing.assert_array_equal(packed[2], a["items"][index])
                if index == ix:
                    break
                step = collect_step(
                    e,
                    boundary,
                    lambda env, tok: proposal_from_action(env, tok, a["raw_u"][index]),
                )
                np.testing.assert_allclose(
                    step.reward.training, a["reward"][index], atol=1e-8, rtol=0
                )
                boundary = step.boundary
            result = dict(
                index=ix,
                scene_id=row["scene_id"],
                provider=provider,
                tick=e.tick,
                selection=selection,
                actions={},
            )
            for name, raw, q_value in [
                ("stored", a["raw_u"][ix], row["min_q"]),
                (
                    "current_mean",
                    np.asarray(row["current_mean_raw_u"]),
                    row["current_mean_q"],
                ),
            ]:
                branch = fork_environment(e)
                step = collect_step(
                    branch,
                    boundary,
                    lambda env, tok: proposal_from_action(env, tok, raw),
                )
                result["actions"][name] = dict(
                    q=q_value,
                    reward=step.reward.training,
                    check_valid=step.result.check_result.valid,
                    reason=step.result.check_result.reason,
                    source=step.result.source.value,
                    bootstrap=bool(step.transition.bootstrap),
                    next_observation_sha256=observation_fingerprint(
                        step.boundary.sac_observation
                    ),
                    outcome=(
                        step.result.terminal_reason.value
                        if step.result.terminal_reason
                        else None
                    ),
                )
                if name == "stored":
                    np.testing.assert_allclose(
                        step.reward.training, a["reward"][ix], atol=1e-8, rtol=0
                    )
            report["action_reproductions"].append(result)
            write_report(args.output, report)
            print(json.dumps(result), flush=True)
    assert versions(args.manifest) == version
    report["status"] = "COMPLETE"
    write_report(args.output, report)
    print(
        json.dumps(
            [dict(updates=m["updates"], groups=m["groups"]) for m in report["models"]],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
