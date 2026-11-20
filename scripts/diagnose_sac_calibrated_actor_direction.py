#!/usr/bin/env python3
"""Probe a hypothetical actor step with old versus evaluated critic bundles.

This is a discarded, read-only counterfactual, NOT a production SAC schedule.
Both copies use the same actor/temperature/optimizers, batch and update RNG.
Conditional rollouts change only the first action and then use the original
frozen stochastic policy. Separate deterministic rollouts use each whole actor.
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
    for name in ("source", "diagnostic", "parent", "states-from", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=4, choices=(4, 8))
    parser.add_argument("--rollout-seed", type=int, default=20261250)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.audit_sac_fixed_policy_continuation import audit
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from scripts.sac_diagnostic_sampling import sample_recent, draw_indices, indices_hash
    from src.contracts.observation import CoreObservation
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.training.sac.collector import collect_step
    from src.training.sac.learner import initialize, update
    from src.training.sac.networks import observation_speed, sample_action
    from src.training.sac.policy import Policy, tensor_observation, proposal_from_action
    from src.training.sac.replay import Replay
    from src.training.sac.runtime import versions, selected_scenes, write_report

    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    checked = audit(args.diagnostic, args.parent, args.source)
    source, diagnostic, origin = (read(p / "report.json") for p in
                                  (args.source, args.diagnostic, args.states_from))
    version = versions(args.manifest)
    assert source["versions"] == diagnostic["versions"] == origin["versions"] == version
    assert diagnostic["status"] == "COMPLETE" and diagnostic["critic_steps"] == 5000
    inputs = dict(checked["source_sha256"])
    for p in (args.states_from / "report.json", args.states_from / "ticks.jsonl", args.manifest):
        inputs[str(p)] = sha(p)
        assert diagnostic["input_sha256"][str(p)] == sha(p)
    sources = {str(p): sha(p) for p in [Path(__file__), *map(Path, (
        "scripts/sac_diagnostic_actor.py", "scripts/sac_diagnostic_critic.py",
        "scripts/sac_diagnostic_sampling.py", "scripts/diagnose_sac_q_values.py",
        "src/training/sac/learner.py", "src/training/sac/networks.py",
        "src/training/sac/policy.py", "src/training/sac/collector.py"))]}
    args.output.mkdir(parents=True)
    report = dict(status="RUNNING", qualified=False, production_checkpoint=False,
        scope="DISCARDED_PAIRED_ACTOR_DIRECTION_PROBE", versions=version,
        input_sha256=inputs, source_sha256=sources, replicates=args.replicates,
        rollout_seed=args.rollout_seed, branches=[], roots=[],
        note="One hypothetical canonical update on copied states, not a valid production "
             "continuation or a new-per-transition learning schedule. Critic bundles include "
             "online/target/optimizer state. Same original actor/temperature/optimizers and "
             "RNG; same one half_recent256 batch from final own replay. No input writes or "
             "learned checkpoint. Two repeated TRAIN roots, not independent evaluation. "
             "Conditional returns exclude first-action entropy; later behavior stays original. "
             "Whole deterministic actor rollouts are separate and not Q calibration truth.")
    output = args.output / "report.json"
    write_report(output, report)
    try:
        gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
        with jax.default_device(gpu):
            template = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                actor=diagnostic_actor(source["arms"]["online"], restoring=True),
                critic=diagnostic_critic(source["arms"]["online"], restoring=True))
            paths = {"old": args.source / "online-learner.msgpack",
                     "evaluated": args.diagnostic / "critic-evaluation-state.msgpack"}
            states = {}
            for name, path in paths.items():
                states[name] = jax.device_put(serialization.from_bytes(template, path.read_bytes()), gpu)
                assert serialization.to_bytes(states[name]) == path.read_bytes()
            original = states["old"]
            for attr in ("actor", "log_alpha", "alpha_optimizer"):
                assert serialization.to_bytes(getattr(original, attr)) == serialization.to_bytes(getattr(states["evaluated"], attr))
            assert int(original.bc_updates) == int(states["evaluated"].bc_updates) == 0
            report["original_update_rng"] = np.asarray(original.rng).tolist()
            report["evaluated_saved_rng"] = np.asarray(states["evaluated"].rng).tolist()
            states["evaluated"] = states["evaluated"].replace(rng=original.rng)
            replay = Replay.load(args.diagnostic / "diagnostic-replay.npz", version)
            assert replay.size == 11000 and not replay.arrays["teacher_accepted"][:replay.size].any()
            before_rng = deepcopy(replay.rng.bit_generator.state)
            rng = np.random.default_rng(); rng.bit_generator.state = deepcopy(before_rng)
            indices = draw_indices(rng, replay.size, "half_recent256")
            batch, batch_hash = sample_recent(replay)
            assert batch_hash == indices_hash(indices) and replay.rng.bit_generator.state == rng.bit_generator.state
            report["batch"] = dict(indices=indices.tolist(), sha256=batch_hash, rng_before=before_rng,
                rng_after=replay.rng.bit_generator.state, terminal=int((~batch["bootstrap"]).sum()),
                new_frozen_experience=int((indices >= 6000).sum()))
            batch = jax.device_put(batch, gpu)
            updates = {}; report["hypothetical_updates"] = {}
            for name, state in states.items():
                updates[name], metrics = update(state, batch, replay.size)
                assert int(updates[name].actor.step) == int(state.actor.step) + 1
                report["hypothetical_updates"][name] = {k: float(v) for k, v in metrics.items()}
            policies = {"original": Policy(original.actor, deterministic=True),
                        **{name: Policy(state.actor, deterministic=True) for name, state in updates.items()}}
            frozen_forward = jax.jit(original.actor.apply_fn)
            alpha = float(jnp.exp(original.log_alpha))
            report["alpha_for_conditional_suffix"] = alpha
            apply1, apply2 = original.q1.apply_fn, original.q2.apply_fn
            @jax.jit
            def value(p1, p2, obs, action):
                return jnp.minimum(apply1({"params": p1}, obs, action), apply2({"params": p2}, obs, action))

        old_rows = [json.loads(line) for line in (args.states_from / "ticks.jsonl").read_text().splitlines()]
        scenes = {(r["scene_id"], r["provider"]): p for r, p in selected_scenes(args.manifest, "train")}
        for case_index, case in enumerate(diagnostic["roots"]):
            old = old_rows[case["source_trace_index"]]
            prefix = [r for r in old_rows if r["lineage"]["kind"] == "main" and
                      r["episode_uid"] == old["episode_uid"] and r["pre_tick"] < case["tick"]]
            assert [r["pre_tick"] for r in prefix] == list(range(case["tick"]))
            with jax.default_device(cpu):
                env = SACPlanningEnvironment()
                boundary = env.reset(parse_scenario(scenes[(case["scene_id"], case["provider"])].read_bytes()),
                                     seed=origin["seed"], provider_spec=case["provider"])
                for row in prefix:
                    step = collect_step(env, boundary, lambda e, t: proposal_from_action(e, t, np.asarray(row["raw_proposal"]["raw_u"])))
                    assert step.transition is not None and step.transition.bootstrap
                    np.testing.assert_allclose(step.reward.training, row["reward"], atol=1e-8, rtol=0)
                    boundary = step.boundary
            assert observation_fingerprint(boundary.sac_observation) == case["observation_sha256"]
            with jax.default_device(gpu):
                means = {name: policy.action(boundary.sac_observation) for name, policy in policies.items()}
            root = {k: case[k] for k in ("scene_id", "provider", "tick", "observation_sha256")}
            root["actions"] = {}
            for name, mu in means.items():
                qs = {}; cross_device = 0.
                with jax.default_device(gpu), jax.default_matmul_precision("highest"):
                    obs = tensor_observation(boundary.sac_observation)
                    for bundle, state in {**states, **{k + "_after_virtual_q": s for k, s in updates.items()}}.items():
                        qs[bundle] = float(value(state.q1.params, state.q2.params, obs, jnp.asarray(mu[None]))[0])
                        with jax.default_device(cpu):
                            cq = float(value(jax.device_put(state.q1.params, cpu), jax.device_put(state.q2.params, cpu),
                                             jax.device_put(obs, cpu), jnp.asarray(mu[None]))[0])
                        difference = abs(cq - qs[bundle])
                        assert difference <= 1e-3, (case_index, name, bundle, difference)
                        cross_device = max(cross_device, difference)
                root["actions"][name] = dict(raw_u=mu.tolist(), q_highest=qs,
                    critic_cpu_gpu_max_difference=cross_device,
                    displacement_from_original=float(np.linalg.norm(mu-means["original"])))
            report["roots"].append(root); write_report(output, report)
            for mode in ("conditional_frozen_suffix", "whole_deterministic_actor"):
                for replicate in range(args.replicates if mode == "conditional_frozen_suffix" else 1):
                    seed = args.rollout_seed + case_index * 100 + replicate
                    for name in means:
                        branch = fork_environment(env); current = boundary
                        parent_token, parent_state, parent_memory = env.token, env._state, env.memory
                        with jax.default_device(gpu): key = jax.random.PRNGKey(seed)
                        trace = []
                        for depth in range(200):
                            ell = 0.
                            with jax.default_device(gpu):
                                if mode == "whole_deterministic_actor":
                                    raw = policies[name].action(current.sac_observation)
                                elif depth == 0:
                                    raw = means[name]
                                else:
                                    key, draw = jax.random.split(key)
                                    obs = tensor_observation(current.sac_observation)
                                    mu, rho = frozen_forward({"params": original.actor.params}, obs)
                                    u, logp, _ = sample_action(mu, rho, observation_speed(obs), draw)
                                    raw, ell = np.asarray(u[0]), float(logp[0])
                            pre = observation_fingerprint(current.sac_observation)
                            with jax.default_device(cpu):
                                step = collect_step(branch, current, lambda e, t: proposal_from_action(e, t, raw))
                            assert step.transition is not None and step.transition.delta_ticks == 1
                            current = step.boundary
                            trace.append(dict(depth=depth, pre_observation_sha256=pre, raw_u=raw.tolist(),
                                reward=step.reward.training, ell=ell, accepted=step.result.check_result.valid,
                                reason=step.result.check_result.reason, source=step.result.source.value,
                                post_speed=branch.ego.speed, bootstrap=bool(step.transition.bootstrap),
                                outcome=step.result.terminal_reason.value if step.result.terminal_reason else None))
                            if not step.transition.bootstrap: break
                        else:
                            raise ValueError("probe failed to reach an actual contract terminal")
                        assert env.token == parent_token and env._state is parent_state and env.memory == parent_memory
                        filename = f"root{case_index}-{mode}-{replicate}-{name}.json"
                        write_report(args.output / filename, dict(status="COMPLETE", rows=trace))
                        result = dict(root=case_index, provider=case["provider"], mode=mode,
                            replicate=replicate, seed=seed, candidate=name, ticks=len(trace),
                            outcome=trace[-1]["outcome"], discounted_reward=sum(.99**r["depth"]*r["reward"] for r in trace),
                            undiscounted_reward=sum(r["reward"] for r in trace),
                            soft_return=(sum(.99**r["depth"]*(r["reward"]-alpha*r["ell"]) for r in trace)
                                         if mode == "conditional_frozen_suffix" else None),
                            rejected=sum(not r["accepted"] for r in trace),
                            braking=sum(r["source"] in ("BRAKE", "EMERGENCY") for r in trace),
                            stationary=sum(r["post_speed"] < .1 for r in trace),
                            trace_file=filename, trace_sha256=sha(args.output / filename))
                        report["branches"].append(result); write_report(output, report)
                        print(json.dumps(result), flush=True)
        for group in (inputs, sources):
            for path, digest in group.items(): assert sha(path) == digest, path
        assert versions(args.manifest) == version
        report.update(status="COMPLETE", original_inputs_unchanged=True, learned_checkpoint_saved=False)
        write_report(output, report)
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_report(output, report)
        raise


if __name__ == "__main__":
    main()
