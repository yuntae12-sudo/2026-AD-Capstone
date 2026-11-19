#!/usr/bin/env python3
"""Read-only train-state TD and actor-direction audit; never saves learned weights."""

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


def episode_suffix(arrays, terminal_index, lookback=10):
    """Require a contiguous complete recorded suffix, never cross episode boundaries."""
    import numpy as np

    ix = terminal_index
    if arrays["bootstrap"][ix] or not (arrays["terminated"][ix] or arrays["truncated"][ix]):
        raise ValueError("requires a recorded terminal boundary")
    rows = np.flatnonzero(arrays["episode_uid"][: ix + 1] == arrays["episode_uid"][ix])
    rows = rows[-(lookback + 1):]
    if len(rows) < 2 or np.any(np.diff(arrays["tick"][rows]) != arrays["delta_ticks"][rows[:-1]]):
        raise ValueError("non-contiguous episode suffix")
    if not arrays["bootstrap"][rows[:-1]].all():
        raise ValueError("intermediate terminal boundary")
    return rows


def recorded_returns(rewards, delta_ticks):
    """Realized discounted rewards of this recorded behavior, NOT current-policy Q*."""
    result = [0.0] * len(rewards)
    value = 0.0
    for k in range(len(rewards) - 1, -1, -1):
        value = float(rewards[k]) + 0.99 ** int(delta_ticks[k]) * value
        result[k] = value
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "manifest", "reference-cases", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--skip-branches", action="store_true", help="Numerical audit only; explicitly mark Waymax branches unexecuted")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 2 <= args.samples <= 128:
        p.error("samples must be 2..128")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from src.contracts.observation import CoreObservation
    from src.contracts.planning import Maneuver
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.learner import initialize, update, targets, draw, gradient_norm
    from src.training.sac.networks import Actor, Critic, normalized_action, observation_speed
    from src.training.sac.policy import tensor_observation, proposal_from_action
    from src.training.sac.replay import Replay, unpack_observation, pack_observation
    from src.training.sac.runtime import versions, write_report, selected_scenes
    from src.training.sac.collector import collect_step
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from scripts.sac_target_components import target_components

    version = versions(args.manifest)
    source = json.loads((args.diagnostic / "report.json").read_text())
    assert source["status"] == "COMPLETE" and source["versions"] == version
    arm = source["arms"]["online"]
    weights = args.diagnostic / "online-learner.msgpack"
    assert file_hash(weights) == arm["learner_sha256"]
    replay_path = args.diagnostic / "replay.npz"
    replay = Replay.load(replay_path, version)
    a = replay.arrays
    assert replay.size == 5000 + source["new_transitions"] and not a["teacher_accepted"][:replay.size].any()
    reference = json.loads(args.reference_cases.read_text())
    assert reference["status"] == "COMPLETE" and reference["versions"] == version
    suffixes = [episode_suffix(a, r["index"]) for r in reference["action_reproductions"]]
    indices = np.concatenate(suffixes)
    returns = np.concatenate([recorded_returns(a["reward"][ix], a["delta_ticks"][ix]) for ix in suffixes])
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    report = dict(status="RUNNING", scope="READ_ONLY_TD_AND_ACTOR_DIRECTION", qualified=False,
                  versions=version, script_sha256=file_hash(__file__), model_sha256=file_hash(weights),
                  target_components_sha256=file_hash(Path(__file__).with_name("sac_target_components.py")),
                  replay_sha256=file_hash(replay_path), reference_sha256=file_hash(args.reference_cases),
                  samples=args.samples, source=str(args.diagnostic), rows=[], branches=[],
                  branches_executed=not args.skip_branches,
                  note="Fixed train failure suffixes; recorded return is behavior-specific, not unbiased current-policy Q. One hypothetical uniform replay update is discarded, no policy checkpoint or replay writes.")
    write_report(args.output, report)
    try:
        with jax.default_device(gpu):
            state = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                               actor=diagnostic_actor(arm, restoring=True),
                               critic=diagnostic_critic(arm, restoring=True))
            state = jax.device_put(serialization.from_bytes(state, weights.read_bytes()), gpu)
            assert int(state.bc_updates) == 0 and int(state.updates) == source["updates"]
            obs = jax.device_put(unpack_observation(a["obs"][indices], a["fields"][indices], a["items"][indices]), gpu)
            nxt = jax.device_put(unpack_observation(a["next_obs"][indices], a["next_fields"][indices], a["next_items"][indices]), gpu)
            raw = jax.device_put(a["raw_u"][indices], gpu)
            def qvalue(q1, q2, observation, action):
                return jnp.minimum(state.q1.apply_fn({"params": q1}, observation, action),
                                   state.q2.apply_fn({"params": q2}, observation, action))
            qvalue = jax.jit(qvalue)
            mu, _ = state.actor.apply_fn({"params": state.actor.params}, obs)
            stored_q = np.asarray(qvalue(state.q1.params, state.q2.params, obs, raw))
            mean_q = np.asarray(qvalue(state.q1.params, state.q2.params, obs, mu))
            mean_actions = np.asarray(mu, np.float64)
            sample_q, target_values, next_q_values, entropy_terms = [], [], [], []
            decomposition_differences = []
            nonterminal = np.flatnonzero(a["bootstrap"][indices])
            compact_next = jax.tree_util.tree_map(lambda x: x[nonterminal], nxt)
            batch_suffix = dict(observation=obs, next_observation=nxt, raw_u=raw,
                                reward=a["reward"][indices], bootstrap=a["bootstrap"][indices], delta_ticks=a["delta_ticks"][indices])
            for k in range(args.samples):
                key = jax.random.PRNGKey(91000 + k)
                u, _ = draw(state.actor, obs, key)
                sample_q.append(np.asarray(qvalue(state.q1.params, state.q2.params, obs, u)))
                actual_target = np.asarray(targets(state, batch_suffix, key))
                target_values.append(actual_target)
                fused_q, fused_entropy, fused_value = target_components(state, compact_next, key)
                next_q = np.zeros(len(indices)); entropy = np.zeros(len(indices))
                next_q[nonterminal] = np.asarray(fused_q)
                entropy[nonterminal] = np.asarray(fused_entropy)
                independent_target = a["reward"][indices] + .99**a["delta_ticks"][indices]*(next_q+entropy)
                decomposition_differences.append(float(np.max(np.abs(actual_target-independent_target))))
                np.testing.assert_allclose(actual_target, independent_target, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(fused_value, fused_q+fused_entropy, atol=1e-10, rtol=1e-10)
                next_q_values.append(next_q); entropy_terms.append(entropy)
            sampled_q, td = np.asarray(sample_q), np.asarray(target_values)
            # Same-state one-step production update, including fresh critics before actor.
            batch = jax.device_put(replay.sample(256), gpu)
            after, metrics = update(state, batch, replay.size)
            _, _, actor_key = jax.random.split(state.rng, 3)
            def loss_terms(params):
                actor = state.actor.replace(params=params)
                u, ell = draw(actor, batch["observation"], actor_key)
                return jnp.stack([jnp.mean(-qvalue(after.q1.params, after.q2.params, batch["observation"], u)),
                                  jnp.mean(jnp.exp(state.log_alpha) * ell)])
            grads = jax.jit(jax.jacrev(loss_terms))(state.actor.params)
            qgrad = jax.tree_util.tree_map(lambda x: x[0], grads)
            egrad = jax.tree_util.tree_map(lambda x: x[1], grads)
            qnorm, enorm = float(gradient_norm(qgrad)), float(gradient_norm(egrad))
            dot = sum(float(jnp.sum(x.astype(jnp.float64) * y)) for x, y in zip(jax.tree_util.tree_leaves(qgrad),jax.tree_util.tree_leaves(egrad)))
            new_mu, _ = after.actor.apply_fn({"params": after.actor.params}, obs)
            new_actions = np.asarray(new_mu, np.float64)
            # Compare actor movement at the SAME critic parameters, not after critic drift.
            moved_q = np.asarray(qvalue(after.q1.params, after.q2.params, obs, new_mu))
            old_mu_new_q = np.asarray(qvalue(after.q1.params, after.q2.params, obs, mu))
            white = np.asarray(normalized_action(mu, observation_speed(obs)))
            moved_white = np.asarray(normalized_action(new_mu, observation_speed(obs)))
            report["actor_direction"] = dict(q_gradient_norm=qnorm, entropy_gradient_norm=enorm,
                cosine=dot/(qnorm*enorm) if qnorm*enorm else None,
                q_mu_gradient_norm=float(gradient_norm(qgrad["mu"])),
                entropy_mu_gradient_norm=float(gradient_norm(egrad["mu"])),
                alpha=float(jnp.exp(state.log_alpha)), batch_size=256,
                batch_terminal=int(np.asarray(batch["bootstrap"]).size-np.asarray(batch["bootstrap"]).sum()),
                update_metrics={k:float(v) for k,v in metrics.items()})
            report["fused_target_decomposition"] = dict(
                max_absolute_difference=max(decomposition_differences),
                per_sample_max_absolute_difference=decomposition_differences,
                tolerance=dict(atol=1e-10,rtol=1e-10),
                note="Fused actor and target-Q graph; checked against production targets. Prior separate-execution discrepancies remain in failed/v2 reports.")
            for k, ix in enumerate(indices):
                row=dict(index=int(ix), scene_id=str(a["scene_id"][ix]), tick=int(a["tick"][ix]),
                         reward=float(a["reward"][ix]), bootstrap=bool(a["bootstrap"][ix]),
                         recorded_suffix_return=float(returns[k]), stored_q=float(stored_q[k]),
                         current_mean_q=float(mean_q[k]), sampled_q_mean=float(sampled_q[:,k].mean()),
                         sampled_q_std=float(sampled_q[:,k].std()), target_mean=float(td[:,k].mean()), target_std=float(td[:,k].std()),
                         next_target_q_mean=float(np.mean(next_q_values,axis=0)[k]),
                         next_entropy_correction_mean=float(np.mean(entropy_terms,axis=0)[k]),
                         mean_q_before_actor_step=float(old_mu_new_q[k]), mean_q_after_actor_step=float(moved_q[k]),
                         normalized_mean_norm=float(np.linalg.norm(white[k])), normalized_mean_movement=float(np.linalg.norm(moved_white[k]-white[k])),
                         raw_mu=mean_actions[k].tolist(), after_mu=new_actions[k].tolist())
                if not row["bootstrap"]:
                    np.testing.assert_array_equal(td[:,k],np.full(args.samples,row["reward"]))
                report["rows"].append(row)
        write_report(args.output, report)
        print(json.dumps(report["actor_direction"]),flush=True)
        # Reconstruct six exact train states and execute recorded/current/updated actions.
        paths={(r["scene_id"],r["provider"]):p for r,p in selected_scenes(args.manifest,"train")}
        by_index={r["index"]:r for r in report["rows"]}
        with jax.default_device(cpu):
            for suffix in ([] if args.skip_branches else suffixes):
                ix=int(suffix[-1]); episode=np.flatnonzero(a["episode_uid"][:replay.size]==a["episode_uid"][ix])
                assert a["tick"][episode[0]] == 0 and episode[-1] == ix
                observation=unpack_observation(a["obs"][ix],a["fields"][ix],a["items"][ix])
                provider="fixed_"+list(Maneuver)[int(np.argmax(observation["features"]["command"][:5]))].value.lower()
                env=SACPlanningEnvironment()
                boundary=env.reset(parse_scenario(paths[(str(a["scene_id"][ix]),provider)].read_bytes()),seed=source["seed"],provider_spec=provider)
                probes=set(int(suffix[-1-offset]) for offset in (0,1,5))
                for index in episode:
                    packed=pack_observation(boundary.sac_observation)
                    np.testing.assert_allclose(packed[0],a["obs"][index],atol=1e-6,rtol=0)
                    np.testing.assert_array_equal(packed[1],a["fields"][index])
                    np.testing.assert_array_equal(packed[2],a["items"][index])
                    if int(index) in probes:
                        row=by_index[int(index)]; case=dict(index=int(index),scene_id=row["scene_id"],provider=provider,tick=int(env.tick),actions={})
                        for name, action in [("stored",a["raw_u"][index]),("mean",np.array(row["raw_mu"])),("after_one_update_mean",np.array(row["after_mu"]))]:
                            branch=fork_environment(env)
                            step=collect_step(branch,boundary,lambda e,t:proposal_from_action(e,t,action))
                            case["actions"][name]=dict(reward=step.reward.training,bootstrap=bool(step.transition.bootstrap),
                                accepted=step.result.check_result.valid,reason=step.result.check_result.reason,source=step.result.source.value,
                                outcome=step.result.terminal_reason.value if step.result.terminal_reason else None,
                                next_observation_sha256=observation_fingerprint(step.boundary.sac_observation))
                            if name=="stored": np.testing.assert_allclose(step.reward.training,a["reward"][index],atol=1e-8,rtol=0)
                        report["branches"].append(case);write_report(args.output,report)
                    step=collect_step(env,boundary,lambda e,t:proposal_from_action(e,t,a["raw_u"][index]))
                    np.testing.assert_allclose(step.reward.training,a["reward"][index],atol=1e-8,rtol=0)
                    boundary=step.boundary
        assert file_hash(weights)==report["model_sha256"] and file_hash(replay_path)==report["replay_sha256"]
        assert versions(args.manifest)==version
        report["status"]="COMPLETE"
        write_report(args.output,report)
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_report(args.output, report)
        raise



if __name__ == "__main__":
    main()
