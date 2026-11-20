#!/usr/bin/env python3
"""Continue a bounded frozen-policy value diagnosis using real self-experience.

The actor/temperature stay fixed; this is NOT coupled SAC training or an
adoptable policy checkpoint. A saved budget boundary is never made terminal.
"""

import argparse
from copy import deepcopy
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
    for name in ("parent", "source", "states-from", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--total-critic-steps", type=int, choices=(1100, 2000, 5000), default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.audit_sac_fixed_policy_critic import audit
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from scripts.sac_diagnostic_sampling import sample_recent, sampling_spec
    from scripts.sac_fixed_policy_continuation import continuation_plan
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
    checked = audit(args.parent, args.source)
    parent, source, origin = (read(p / "report.json") for p in (args.parent, args.source, args.states_from))
    version = versions(args.manifest)
    assert parent["versions"] == source["versions"] == origin["versions"] == version
    tick_prefix = (args.parent / "ticks.jsonl").read_bytes()
    update_prefix = (args.parent / "updates.jsonl").read_bytes()
    assert tick_prefix.endswith(b"\n") and update_prefix.endswith(b"\n")
    last = json.loads(tick_prefix.splitlines()[-1])
    replay_path = args.parent / "diagnostic-replay.npz"
    with np.load(replay_path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
    plan = continuation_plan(parent, last, meta, args.total_critic_steps)
    replay = Replay.load(replay_path, version)
    inherited = plan["inherited_replay_size"]
    assert replay.size == replay.cursor == inherited
    assert not replay.arrays["teacher_accepted"][:inherited].any()
    prefix_hashes = {k: digest(v[:inherited].tobytes()) for k, v in replay.arrays.items()}
    weights = args.parent / "critic-evaluation-state.msgpack"
    inputs = dict(parent["input_sha256"], **checked["source_sha256"])
    for p in (args.states_from / "report.json", args.states_from / "ticks.jsonl", args.manifest):
        inputs[str(p)] = file_hash(p)
    # Confirm that the supplied root source is the exact one used by the parent.
    normalized = {Path(p).resolve(): h for p, h in parent["input_sha256"].items()}
    for p in (args.states_from / "report.json", args.states_from / "ticks.jsonl", args.manifest):
        assert normalized[p.resolve()] == file_hash(p), p
    paths = [Path(__file__), Path("scripts/sac_fixed_policy_continuation.py"),
             Path("scripts/audit_sac_fixed_policy_critic.py")]
    sources = dict(parent["source_sha256"], **{str(p): file_hash(p) for p in paths})
    args.output.mkdir(parents=True)
    report = dict(status="RUNNING", qualified=False, production_checkpoint=False,
        scope="FIXED_POLICY_CRITIC_CONTINUATION_NOT_SAC_TRAINING", versions=version,
        plan=plan, parent_directory=str(args.parent.resolve()), source_directory=str(args.source.resolve()),
        source_sha256=sources, input_sha256=inputs, source_model_sha256=parent["source_model_sha256"],
        inherited_state_sha256=file_hash(weights), inherited_prefix_sha256=prefix_hashes,
        inherited_ticks_sha256=digest(tick_prefix), inherited_updates_sha256=digest(update_prefix),
        inherited_ticks_bytes=len(tick_prefix), inherited_updates_bytes=len(update_prefix),
        inherited_replay_size=inherited, inherited_replay_rng=deepcopy(meta["rng"]),
        replay_sampling=sampling_spec("half_recent256"), collection_seed=plan["collection_seed"],
        budget=plan["total_critic_steps"], critic_steps=1000, actual_transitions=1000,
        new_critic_steps=0, new_actual_transitions=0, actor_updates=0, temperature_updates=0,
        roots=deepcopy(parent["roots"]), episodes=deepcopy(parent["episodes"]),
        probes=deepcopy(parent["probes"]), snapshots=[],
        note="Bounded continuation of fixed-policy critic diagnosis, not coupled SAC training. "
             "Actual self-experience; one canonical Q update per new transition. Actor/alpha fixed. "
             "Saved critic/targets/optimizers/learner and replay RNG restored exactly. New rollouts start "
             "at the same two TRAIN roots with unused seeds, not an exact simulator resume. "
             "Inherited budget bootstrap preserved. No policy promotion, teacher, or automatic extension.")
    output = args.output / "report.json"
    write_report(output, report)
    try:
        cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
        with jax.default_device(gpu):
            template = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                actor=diagnostic_actor(source["arms"]["online"], restoring=True),
                critic=diagnostic_critic(source["arms"]["online"], restoring=True))
            state = jax.device_put(serialization.from_bytes(template, weights.read_bytes()), gpu)
            assert serialization.to_bytes(state) == weights.read_bytes()
            assert int(state.updates) == source["updates"] + 1000 and int(state.bc_updates) == 0
            initial_actor = serialization.to_bytes(state.actor)
            initial_temperature = serialization.to_bytes((state.log_alpha, state.alpha_optimizer))
            assert digest(initial_actor) == parent["actor_sha256"]
            assert digest(initial_temperature) == parent["temperature_sha256"]
            report.update(state_restore_bytes_identical=True, actor_sha256=digest(initial_actor),
                          temperature_sha256=digest(initial_temperature), alpha=float(jnp.exp(state.log_alpha)))
            assert report["alpha"] == parent["alpha"]
            apply1, apply2 = state.q1.apply_fn, state.q2.apply_fn

            @jax.jit
            def values(p1, p2, observation, raw):
                return jnp.minimum(apply1({"params": p1}, observation, raw),
                                   apply2({"params": p2}, observation, raw))

        old_rows = [json.loads(line) for line in (args.states_from / "ticks.jsonl").read_text().splitlines()]
        scenes = {(r["scene_id"], r["provider"]): p for r, p in selected_scenes(args.manifest, "train")}
        roots = []
        for case in parent["roots"]:
            row = old_rows[case["source_trace_index"]]
            assert row["lineage"]["kind"] == "main" and row["check_valid"]
            assert (row["scene_id"], row["provider"], row["pre_tick"]) == (case["scene_id"], case["provider"], case["tick"])
            prefix = [r for r in old_rows if r["lineage"]["kind"] == "main"
                      and r["episode_uid"] == row["episode_uid"] and r["pre_tick"] < case["tick"]]
            assert [r["pre_tick"] for r in prefix] == list(range(case["tick"]))
            with jax.default_device(cpu):
                env = SACPlanningEnvironment()
                boundary = env.reset(parse_scenario(scenes[(case["scene_id"], case["provider"])].read_bytes()),
                                     seed=origin["seed"], provider_spec=case["provider"])
                for old in prefix:
                    raw = np.asarray(old["raw_proposal"]["raw_u"], np.float64)
                    step = collect_step(env, boundary, lambda e, t: proposal_from_action(e, t, raw))
                    assert step.transition is not None and step.transition.bootstrap
                    np.testing.assert_allclose(step.reward.training, old["reward"], rtol=0, atol=1e-8)
                    boundary = step.boundary
            assert observation_fingerprint(boundary.sac_observation) == case["observation_sha256"]
            roots.append((env, boundary, np.asarray(case["reference_mean_raw_u"], np.float64), case))

        def probe():
            result = dict(critic_steps=report["critic_steps"], cases=[])
            assert serialization.to_bytes(state.actor) == initial_actor
            assert serialization.to_bytes((state.log_alpha, state.alpha_optimizer)) == initial_temperature
            for _, boundary, mean, case in roots:
                with jax.default_device(gpu):
                    observation = tensor_observation(boundary.sac_observation)
                    raw = jnp.asarray(mean[None])
                    q = float(np.asarray(values(state.q1.params, state.q2.params, observation, raw))[0])
                    target = float(np.asarray(values(state.target1, state.target2, observation, raw))[0])
                result["cases"].append(dict(provider=case["provider"], scene_id=case["scene_id"], q=q, target_q=target,
                    q_minus_reference_mean=q-case["reference_summary"]["mean_soft_return"]))
            return result

        report["resume_probe"] = probe()
        for new, old in zip(report["resume_probe"]["cases"], parent["probes"][-1]["cases"]):
            assert new["provider"] == old["provider"]
            for key in ("q", "target_q"):
                assert abs(new[key] - old[key]) <= 1e-3, (key, new, old)
        report["resume_probe_tolerance"] = 1e-3
        write_report(output, report)
        (args.output / "ticks.jsonl").write_bytes(tick_prefix)
        (args.output / "updates.jsonl").write_bytes(update_prefix)
        rollout = plan["first_new_rollout"]
        with (args.output / "ticks.jsonl").open("a") as ticks, (args.output / "updates.jsonl").open("a") as updates:
            while report["critic_steps"] < plan["total_critic_steps"]:
                root_index = rollout % 2
                env, boundary, _, case = roots[root_index]
                original_state, original_memory, original_token = env._state, env.memory, env.token
                branch, current = fork_environment(env), boundary
                policy_seed = plan["collection_seed"] + rollout
                with jax.default_device(gpu):
                    policy = Policy(state.actor, seed=policy_seed)
                episode = dict(rollout=rollout, root_index=root_index, policy_seed=policy_seed,
                               scene_id=case["scene_id"], provider=case["provider"], ticks=0,
                               accepted=0, return_training=0., outcome="IN_PROGRESS")
                report["episodes"].append(episode)
                for depth in range(200):
                    if report["critic_steps"] >= plan["total_critic_steps"]:
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
                    count = report["critic_steps"] + 1
                    report.update(critic_steps=count, actual_transitions=count,
                                  new_critic_steps=count-1000, new_actual_transitions=count-1000)
                    assert replay.size == 6000 + count and int(state.updates) == source["updates"] + count
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
                    if count % 100 == 0 or count == plan["total_critic_steps"]:
                        report["probes"].append(probe())
                    if count % 1000 == 0 or count == plan["total_critic_steps"]:
                        path = args.output / f"critic-evaluation-{count}.msgpack"
                        path.write_bytes(serialization.to_bytes(state))
                        report["snapshots"].append(dict(critic_steps=count, state_file=path.name, sha256=file_hash(path)))
                    write_report(output, report)
                    if not step.transition.bootstrap:
                        break
                else:
                    raise RuntimeError("actual terminal not reached within200 ticks")
                if episode["outcome"] == "IN_PROGRESS":
                    assert report["critic_steps"] == plan["total_critic_steps"]
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
        assert (args.output / "ticks.jsonl").read_bytes().startswith(tick_prefix)
        assert (args.output / "updates.jsonl").read_bytes().startswith(update_prefix)
        for group in ("source_sha256", "input_sha256"):
            for path, expected in report[group].items():
                assert file_hash(path) == expected, path
        assert versions(args.manifest) == version
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
