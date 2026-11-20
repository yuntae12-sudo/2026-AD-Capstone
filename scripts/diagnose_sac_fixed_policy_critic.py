#!/usr/bin/env python3
"""Evaluate critic learning on new real experience from one frozen policy.

This deliberately freezes actor/temperature and is NOT coupled SAC training.
Original inputs remain immutable; no production policy checkpoint is emitted.
"""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "states-from", "returns", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(100, 1000), default=1000)
    parser.add_argument("--collection-seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.audit_sac_branch_replay import audit
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from scripts.sac_diagnostic_sampling import report_sampling_mode, sample_recent, sampling_spec
    from scripts.sac_fixed_policy_critic import critic_evaluation_step
    from src.contracts.observation import CoreObservation
    from src.contracts.planning import to_wire
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.collector import collect_step
    from src.training.sac.learner import initialize
    from src.training.sac.policy import Policy, proposal_from_action, tensor_observation
    from src.training.sac.replay import Replay
    from src.training.sac.runtime import selected_scenes, versions, write_report

    read = lambda p: json.loads(p.read_text())
    digest = lambda value: hashlib.sha256(value).hexdigest()
    source = read(args.diagnostic / "report.json")
    origin = read(args.states_from / "report.json")
    reference = read(args.returns / "report.json")
    version = versions(args.manifest)
    assert source["status"] == origin["status"] == reference["status"] == "COMPLETE"
    assert source["versions"] == origin["versions"] == reference["versions"] == version
    assert report_sampling_mode(source) == "half_recent256"
    assert reference["source_diagnostic_sha256"] == file_hash(args.diagnostic / "report.json")
    assert reference["states_source_sha256"] == file_hash(args.states_from / "report.json")
    assert reference["states_replay_sha256"] == file_hash(args.states_from / "replay.npz")
    weights = args.diagnostic / "online-learner.msgpack"
    assert reference["model_sha256"] == source["arms"]["online"]["learner_sha256"] == file_hash(weights)
    assert reference["replay_sha256"] == file_hash(args.diagnostic / "replay.npz")
    assert len(reference["cases"]) == 2 and reference["replicates"] == 4
    assert {c["provider"] for c in reference["cases"]} == {"fixed_keep", "fixed_follow"}
    audit(args.diagnostic, args.manifest)
    audit(args.states_from, args.manifest)
    replay = Replay.load(args.diagnostic / "replay.npz", version)
    inherited = replay.size
    assert inherited == 6000 and replay.cursor == inherited
    assert inherited + args.steps < replay.capacity
    assert not replay.arrays["teacher_accepted"][:inherited].any()
    prefix_hashes = {key: digest(value[:inherited].tobytes()) for key, value in replay.arrays.items()}
    initial_replay_rng = replay.rng.bit_generator.state
    inputs = [args.diagnostic / "report.json", weights, args.diagnostic / "replay.npz",
              args.states_from / "report.json", args.states_from / "replay.npz",
              args.states_from / "ticks.jsonl", args.returns / "report.json", args.manifest]
    for case in reference["cases"]:
        assert len(case["rollouts"]) == 4
        for rollout in case["rollouts"]:
            path = args.returns / rollout["trace_file"]
            assert path.resolve().parent == args.returns.resolve()
            assert file_hash(path) == rollout["trace_sha256"]
            inputs.append(path)
    paths = [Path(__file__), Path(__file__).with_name("sac_fixed_policy_critic.py"),
             Path(__file__).with_name("sac_diagnostic_sampling.py"),
             Path(__file__).with_name("sac_diagnostic_actor.py"),
             Path(__file__).with_name("sac_diagnostic_critic.py"),
             Path(__file__).with_name("sac_physical_critic.py"),
             Path(__file__).with_name("diagnose_sac_q_values.py"),
             Path("src/training/sac/learner.py"), Path("src/training/sac/policy.py"),
             Path("src/training/sac/replay.py"), Path("src/training/sac/collector.py")]
    args.output.mkdir(parents=True)
    report = dict(status="RUNNING", qualified=False, production_checkpoint=False,
                  scope="FIXED_POLICY_CRITIC_EVALUATION_NOT_SAC_TRAINING",
                  versions=version, source_sha256={str(p): file_hash(p) for p in paths},
                  input_sha256={str(p): file_hash(p) for p in inputs},
                  source_model_sha256=file_hash(weights), budget=args.steps,
                  inherited_replay_size=inherited, inherited_replay_rng=initial_replay_rng,
                  inherited_prefix_sha256=prefix_hashes, replay_sampling=sampling_spec("half_recent256"),
                  collection_seed=args.collection_seed, critic_steps=0, actual_transitions=0,
                  actor_updates=0, temperature_updates=0, roots=[], episodes=[], probes=[],
                  note="New actual Gaussian self-experience at two fixed TRAIN roots, not validation. "
                       "Canonical SAC Q/target update once per transition; actor/temperature updates discarded. "
                       "This is value diagnosis, not a contract-compliant coupled SAC training candidate. "
                       "Original model/replay unchanged. MC reference has four samples per root, not exact Q truth. "
                       "No policy promotion, teacher, fabricated terminal, or automatic budget extension.")
    output = args.output / "report.json"
    write_report(output, report)
    try:
        cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
        with jax.default_device(gpu):
            state = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                               actor=diagnostic_actor(source["arms"]["online"], restoring=True),
                               critic=diagnostic_critic(source["arms"]["online"], restoring=True))
            state = jax.device_put(serialization.from_bytes(state, weights.read_bytes()), gpu)
            assert serialization.to_bytes(state) == weights.read_bytes()
            assert int(state.updates) == source["updates"] and int(state.bc_updates) == 0
            initial_actor = serialization.to_bytes(state.actor)
            initial_temperature = serialization.to_bytes((state.log_alpha, state.alpha_optimizer))
            report["actor_sha256"] = digest(initial_actor)
            report["temperature_sha256"] = digest(initial_temperature)
            report["alpha"] = float(jnp.exp(state.log_alpha))
            assert report["alpha"] == reference["alpha"]
            mean_policy = Policy(state.actor, deterministic=True)
            apply1, apply2 = state.q1.apply_fn, state.q2.apply_fn

            @jax.jit
            def values(params1, params2, observation, raw):
                return jnp.minimum(apply1({"params": params1}, observation, raw),
                                   apply2({"params": params2}, observation, raw))

        rows = [json.loads(line) for line in (args.states_from / "ticks.jsonl").read_text().splitlines()]
        scenes = {(r["scene_id"], r["provider"]): p for r, p in selected_scenes(args.manifest, "train")}
        roots = []
        for case in reference["cases"]:
            row = rows[case["source_trace_index"]]
            assert row["lineage"]["kind"] == "main" and row["check_valid"]
            assert (row["scene_id"], row["provider"], row["pre_tick"]) == (
                case["scene_id"], case["provider"], case["tick"])
            prefix = [r for r in rows if r["lineage"]["kind"] == "main"
                      and r["episode_uid"] == row["episode_uid"] and r["pre_tick"] < case["tick"]]
            assert [r["pre_tick"] for r in prefix] == list(range(case["tick"]))
            with jax.default_device(cpu):
                env = SACPlanningEnvironment()
                boundary = env.reset(parse_scenario(scenes[(case["scene_id"], case["provider"])].read_bytes()),
                                     seed=origin["seed"], provider_spec=case["provider"])
                for parent in prefix:
                    raw = np.asarray(parent["raw_proposal"]["raw_u"], np.float64)
                    step = collect_step(env, boundary, lambda e, t: proposal_from_action(e, t, raw))
                    assert step.transition is not None and step.transition.bootstrap
                    np.testing.assert_allclose(step.reward.training, parent["reward"], rtol=0, atol=1e-8)
                    boundary = step.boundary
            assert observation_fingerprint(boundary.sac_observation) == case["observation_sha256"]
            with jax.default_device(gpu):
                mean = mean_policy.action(boundary.sac_observation)
            reference_mean = np.asarray(case["mean_raw_u"], np.float64)
            difference = float(np.max(np.abs(mean - reference_mean)))
            assert difference <= 1e-9
            roots.append((env, boundary, reference_mean, case))
            report["roots"].append(dict(scene_id=case["scene_id"], provider=case["provider"],
                tick=case["tick"], observation_sha256=case["observation_sha256"],
                source_trace_index=case["source_trace_index"], mean_forward_max_difference=difference,
                reference_mean_raw_u=case["mean_raw_u"], reference_summary=case["summary"]))

        def probe():
            result = dict(critic_steps=report["critic_steps"], cases=[])
            for _, boundary, mean, case in roots:
                with jax.default_device(gpu):
                    q = float(np.asarray(values(state.q1.params, state.q2.params,
                        tensor_observation(boundary.sac_observation), jnp.asarray(mean[None])))[0])
                    target = float(np.asarray(values(state.target1, state.target2,
                        tensor_observation(boundary.sac_observation), jnp.asarray(mean[None])))[0])
                if report["critic_steps"] == 0:
                    assert abs(q - case["mean_first_action_q"]) <= 1e-3
                result["cases"].append(dict(provider=case["provider"], scene_id=case["scene_id"],
                    q=q, target_q=target, q_minus_reference_mean=q-case["summary"]["mean_soft_return"]))
            report["probes"].append(result)

        probe()
        rollout = 0
        with (args.output / "ticks.jsonl").open("x") as ticks, (args.output / "updates.jsonl").open("x") as updates:
            while report["critic_steps"] < args.steps:
                root_index = rollout % len(roots)
                env, boundary, _, case = roots[root_index]
                original_state, original_memory, original_token = env._state, env.memory, env.token
                branch, current = fork_environment(env), boundary
                policy_seed = args.collection_seed + rollout
                with jax.default_device(gpu):
                    policy = Policy(state.actor, seed=policy_seed)
                episode = dict(rollout=rollout, root_index=root_index, policy_seed=policy_seed,
                               scene_id=case["scene_id"], provider=case["provider"], ticks=0,
                               accepted=0, return_training=0., outcome="IN_PROGRESS")
                report["episodes"].append(episode)
                for depth in range(200):
                    if report["critic_steps"] >= args.steps:
                        episode["outcome"] = "DIAGNOSTIC_BUDGET"
                        break
                    before = observation_fingerprint(current.sac_observation)
                    with jax.default_device(gpu):
                        raw = policy.action(current.sac_observation)
                    with jax.default_device(cpu):
                        step = collect_step(branch, current, lambda e, t: proposal_from_action(e, t, raw), replay)
                    assert step.transition is not None and step.transition.delta_ticks == 1
                    result = step.result
                    assert step.transition.bootstrap == (not (result.terminated or result.truncated))
                    batch, batch_hash = sample_recent(replay)
                    old_actor = state.actor
                    with jax.default_device(gpu):
                        state, metrics = critic_evaluation_step(state, batch, replay.size)
                    assert state.actor is old_actor
                    report["critic_steps"] += 1
                    report["actual_transitions"] += 1
                    assert replay.size == inherited + report["actual_transitions"]
                    assert int(state.updates) == source["updates"] + report["critic_steps"]
                    count = report["critic_steps"]
                    ticks.write(json.dumps(dict(collection_id=count, rollout=rollout, root_index=root_index,
                        policy_seed=policy_seed, branch_step=depth, scene_id=case["scene_id"], provider=case["provider"],
                        pre_tick=result.pre_token.sim_tick, episode_uid=result.pre_token.episode_uid,
                        observation_sha256=before, next_observation_sha256=observation_fingerprint(step.boundary.sac_observation),
                        raw_proposal=to_wire(result.raw_proposal), check_valid=result.check_result.valid,
                        check_reason=result.check_result.reason, source=result.source.value,
                        actual_control=to_wire(result.actual_control), reward=step.reward.training,
                        bootstrap=step.transition.bootstrap, delta_ticks=step.transition.delta_ticks,
                        terminated=result.terminated, truncated=result.truncated,
                        outcome=result.terminal_reason.value if result.terminal_reason else None), allow_nan=False) + "\n")
                    updates.write(json.dumps(dict(critic_steps=count, replay_size=replay.size,
                        replay_indices_sha256=batch_hash, actor_update_discarded=True, temperature_update_discarded=True,
                        canonical_hypothetical_metrics={k: float(v) for k, v in metrics.items()}), allow_nan=False) + "\n")
                    ticks.flush()
                    updates.flush()
                    episode["ticks"] += 1
                    episode["accepted"] += int(result.check_result.valid)
                    episode["return_training"] += step.reward.training
                    current = step.boundary
                    if not step.transition.bootstrap:
                        episode["outcome"] = result.terminal_reason.value
                    if count % 100 == 0 or count == args.steps:
                        probe()
                    write_report(output, report)
                    if not step.transition.bootstrap:
                        break
                else:
                    raise RuntimeError("actual terminal not reached within200 ticks")
                if episode["outcome"] == "IN_PROGRESS":
                    assert report["critic_steps"] == args.steps
                    episode["outcome"] = "DIAGNOSTIC_BUDGET"
                assert env._state is original_state and env.memory == original_memory and env.token == original_token
                assert observation_fingerprint(env._boundary().sac_observation) == case["observation_sha256"]
                rollout += 1
                print(json.dumps(episode), flush=True)
                write_report(output, report)
        assert serialization.to_bytes(state.actor) == initial_actor
        assert serialization.to_bytes((state.log_alpha, state.alpha_optimizer)) == initial_temperature
        assert int(state.bc_updates) == 0 and not replay.arrays["teacher_accepted"][:replay.size].any()
        for key, value in replay.arrays.items():
            assert digest(value[:inherited].tobytes()) == prefix_hashes[key], key
        for group in ("source_sha256", "input_sha256"):
            for path, expected in report[group].items():
                assert file_hash(path) == expected, path
        assert versions(args.manifest) == version
        # Deliberately distinct names; this is not a production/SAC checkpoint.
        state_path = args.output / "critic-evaluation-state.msgpack"
        state_path.write_bytes(serialization.to_bytes(state))
        replay.save(args.output / "diagnostic-replay.npz")
        report.update(status="COMPLETE", actor_and_temperature_bytes_unchanged=True,
                      original_inputs_unchanged=True, inherited_prefix_arrays_unchanged=len(prefix_hashes),
                      evaluation_state_sha256=file_hash(state_path),
                      diagnostic_replay_sha256=file_hash(args.output / "diagnostic-replay.npz"),
                      final_replay_size=replay.size, final_replay_rng=replay.rng.bit_generator.state)
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(output, report)


if __name__ == "__main__":
    main()
