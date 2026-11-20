#!/usr/bin/env python3
"""Localize Q optimism on saved frozen-policy paths, without new learning.

Replay the recorded raw actions and verify every outcome and next-observation
hash. Evaluate online/target critics on that same path and telescope the gap.
The suffix is a recorded stochastic sample, not an expected Bellman target.
"""

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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "states-from", "returns", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.sac_bellman_trace import decompose
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from src.contracts.observation import CoreObservation
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.collector import collect_step
    from src.training.sac.learner import initialize
    from src.training.sac.policy import tensor_observation, proposal_from_action
    from src.training.sac.runtime import versions, selected_scenes, write_report

    read = lambda p: json.loads(p.read_text())
    source = read(args.diagnostic / "report.json")
    origin = read(args.states_from / "report.json")
    returns = read(args.returns / "report.json")
    version = versions(args.manifest)
    assert source["status"] == origin["status"] == returns["status"] == "COMPLETE"
    assert source["versions"] == origin["versions"] == returns["versions"] == version
    assert returns["source_diagnostic_sha256"] == file_hash(args.diagnostic / "report.json")
    assert returns["states_source_sha256"] == file_hash(args.states_from / "report.json")
    assert returns["states_replay_sha256"] == file_hash(args.states_from / "replay.npz")
    weights = args.diagnostic / "online-learner.msgpack"
    assert returns["model_sha256"] == source["arms"]["online"]["learner_sha256"] == file_hash(weights)
    assert returns["replay_sha256"] == file_hash(args.diagnostic / "replay.npz")
    assert len(returns["cases"]) == 2 and returns["replicates"] == 4
    assert {c["provider"] for c in returns["cases"]} == {"fixed_keep", "fixed_follow"}
    rows = [json.loads(line) for line in (args.states_from / "ticks.jsonl").read_text().splitlines()]
    inputs = [args.diagnostic / "report.json", weights, args.diagnostic / "replay.npz",
              args.states_from / "report.json", args.states_from / "replay.npz",
              args.states_from / "ticks.jsonl", args.returns / "report.json", args.manifest]
    for case in returns["cases"]:
        assert len(case["rollouts"]) == 4
        for rollout in case["rollouts"]:
            path = args.returns / rollout["trace_file"]
            assert path.resolve().parent == args.returns.resolve()
            assert file_hash(path) == rollout["trace_sha256"]
            inputs.append(path)
    source_paths = [Path(__file__), Path(__file__).with_name("sac_bellman_trace.py"),
                    Path(__file__).with_name("sac_diagnostic_actor.py"),
                    Path(__file__).with_name("sac_diagnostic_critic.py"),
                    Path(__file__).with_name("sac_physical_critic.py"),
                    Path(__file__).with_name("diagnose_sac_q_values.py")]
    args.output.mkdir(parents=True)
    report = dict(status="RUNNING", qualified=False, scope="SAVED_POLICY_PATH_BELLMAN_ACCOUNTING",
                  versions=version, input_sha256={str(p): file_hash(p) for p in inputs},
                  source_sha256={str(p): file_hash(p) for p in source_paths},
                  model_sha256=file_hash(weights), cases=[],
                  note="No new actions, replay writes or learner updates. Replays saved train paths, not new independent episodes. "
                       "One-sample next actions and sampled returns are not exact expected Q targets. "
                       "Target-lag contribution is an accounting term, not proof that changing tau would improve driving.")
    output = args.output / "report.json"
    write_report(output, report)
    try:
        cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
        with jax.default_device(gpu):
            arm = source["arms"]["online"]
            state = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                               actor=diagnostic_actor(arm, restoring=True),
                               critic=diagnostic_critic(arm, restoring=True))
            state = jax.device_put(serialization.from_bytes(state, weights.read_bytes()), gpu)
            assert int(state.updates) == source["updates"] and int(state.bc_updates) == 0
            alpha = float(jnp.exp(state.log_alpha))
            assert alpha == returns["alpha"]

            @jax.jit
            def values(observation, raw):
                online = jnp.minimum(state.q1.apply_fn({"params": state.q1.params}, observation, raw),
                                     state.q2.apply_fn({"params": state.q2.params}, observation, raw))
                target = jnp.minimum(state.q1.apply_fn({"params": state.target1}, observation, raw),
                                     state.q2.apply_fn({"params": state.target2}, observation, raw))
                return online, target

        scenes = {(r["scene_id"], r["provider"]): p for r, p in selected_scenes(args.manifest, "train")}
        for case in returns["cases"]:
            root_row = rows[case["source_trace_index"]]
            assert root_row["lineage"]["kind"] == "main" and root_row["check_valid"]
            assert (root_row["scene_id"], root_row["provider"], root_row["pre_tick"]) == (
                case["scene_id"], case["provider"], case["tick"])
            prefix = [r for r in rows if r["lineage"]["kind"] == "main"
                      and r["episode_uid"] == root_row["episode_uid"] and r["pre_tick"] < case["tick"]]
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
            original_state, original_memory, original_token = env._state, env.memory, env.token
            result_case = dict(scene_id=case["scene_id"], provider=case["provider"], tick=case["tick"], rollouts=[])
            report["cases"].append(result_case)
            for rollout in case["rollouts"]:
                trace = [json.loads(line) for line in (args.returns / rollout["trace_file"]).read_text().splitlines()]
                assert len(trace) == rollout["ticks"] and trace[-1]["outcome"] == rollout["outcome"]
                np.testing.assert_array_equal(trace[0]["raw_u"], case["mean_raw_u"])
                branch, current = fork_environment(env), boundary
                qs, targets = [], []
                for index, record in enumerate(trace):
                    assert record["index"] == index and record["pre_tick"] == branch.tick
                    raw = np.asarray(record["raw_u"], np.float64)
                    with jax.default_device(gpu):
                        q, tq = values(tensor_observation(current.sac_observation), jnp.asarray(raw[None]))
                        qs.append(float(np.asarray(q)[0]))
                        targets.append(float(np.asarray(tq)[0]))
                    with jax.default_device(cpu):
                        step = collect_step(branch, current, lambda e, t: proposal_from_action(e, t, raw))
                    assert step.transition is not None
                    assert step.transition.delta_ticks == 1
                    np.testing.assert_allclose(step.reward.training, record["reward"], rtol=0, atol=1e-8)
                    assert step.transition.bootstrap == record["bootstrap"]
                    assert step.result.check_result.valid == record["check_valid"]
                    assert step.result.check_result.reason == record["check_reason"]
                    assert step.result.source.value == record["source"]
                    outcome = step.result.terminal_reason.value if step.result.terminal_reason else None
                    assert outcome == record["outcome"]
                    assert observation_fingerprint(step.boundary.sac_observation) == record["next_observation_sha256"]
                    current = step.boundary
                assert env._state is original_state and env.memory == original_memory and env.token == original_token
                assert observation_fingerprint(env._boundary().sac_observation) == case["observation_sha256"]
                result = decompose([x["reward"] for x in trace], [x["ell"] for x in trace], qs, targets,
                                   [x["bootstrap"] for x in trace], alpha)
                np.testing.assert_allclose(result["summary"]["soft_return"], rollout["soft_return"], rtol=0, atol=1e-10)
                # Forward graph fusion can perturb float32 critics slightly.
                np.testing.assert_allclose(qs[0], case["mean_first_action_q"], rtol=0, atol=1e-3)
                for measured, record in zip(result["rows"], trace):
                    measured.update({k: record[k] for k in ("pre_tick", "reward", "ell", "bootstrap", "source", "check_reason", "outcome")})
                result["summary"]["q0_difference_from_original_forward"] = qs[0] - case["mean_first_action_q"]
                name = f"{case['provider']}-rep{rollout['replicate']}.json"
                write_report(args.output / name, result)
                item = dict(replicate=rollout["replicate"], seed=rollout["seed"], outcome=rollout["outcome"],
                            **result["summary"], file=name, sha256=file_hash(args.output / name),
                            largest_positive_residuals=sorted(result["rows"], key=lambda x: x["discounted_residual"], reverse=True)[:5])
                result_case["rollouts"].append(item)
                write_report(output, report)
                print(json.dumps(dict(provider=case["provider"], replicate=rollout["replicate"], **result["summary"])), flush=True)
            summary_keys = ("initial_gap", "discounted_target_residual_sum", "discounted_target_lag_sum",
                            "terminal_contribution", "nonterminal_contribution")
            result_case["mean_components"] = {k: float(np.mean([r[k] for r in result_case["rollouts"]])) for k in summary_keys}
            write_report(output, report)
        assert serialization.to_bytes(state) == weights.read_bytes()
        assert versions(args.manifest) == version
        for group in ("input_sha256", "source_sha256"):
            assert all(file_hash(Path(p)) == digest for p, digest in report[group].items())
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(output, report)


if __name__ == "__main__":
    main()
