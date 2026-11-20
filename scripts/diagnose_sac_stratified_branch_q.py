#!/usr/bin/env python3
"""Compare a frozen policy at independently audited historical train branch failures."""

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


def select_cases(rows):
    """First actually terminal branch per provider, selected without any Q values."""
    selected = {}
    for i, row in enumerate(rows):
        if (row["lineage"]["kind"] == "failure_branch"
                and row["terminal_reason"] in ("COLLISION", "OFFROAD")
                and (row["terminated"] or row["truncated"])):
            selected.setdefault(row["provider"], i)
    if not selected:
        raise ValueError("no actual terminal branch cases")
    if not set(selected) <= {"fixed_keep", "fixed_follow"}:
        raise ValueError("initial diagnostic is KEEP/FOLLOW only")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "states-from", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.samples <= 32:
        parser.error("samples must be 1..32")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from src.contracts.observation import CoreObservation
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.learner import initialize
    from src.training.sac.networks import Actor
    from src.training.sac.policy import tensor_observation, proposal_from_action, Policy
    from src.training.sac.replay import Replay, pack_observation
    from src.training.sac.collector import collect_step
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from scripts.sac_diagnostic_critic import diagnostic_critic, critic_factory_hash
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from scripts.sac_stratified_diagnostic_audit import audit
    from scripts.sac_branch_reconstruction import branch_history_indices

    version = versions(args.manifest)
    source = json.loads((args.diagnostic / "report.json").read_text())
    assert source["status"] == "COMPLETE" and source["versions"] == version
    model_check = audit(args.diagnostic, args.manifest)
    origin = json.loads((args.states_from / "report.json").read_text())
    assert origin["status"] == "COMPLETE" and origin["versions"] == version
    storage_check = audit(args.states_from, args.manifest)
    rows = [json.loads(line) for line in (args.states_from / "ticks.jsonl").read_text().splitlines()]
    selected = select_cases(rows)
    replay = Replay.load(args.states_from / "replay.npz", version)
    a = replay.arrays
    arm = source["arms"]["online"]
    weights = args.diagnostic / "online-learner.msgpack"
    assert file_hash(weights) == arm["learner_sha256"]
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    report = dict(status="RUNNING", qualified=False,
                  scope="ACTUAL_CURRENT_POLICY_AT_COLLECTED_TRAIN_BRANCH_FAILURES",
                  versions=version, script_sha256=file_hash(__file__),
                  reconstruction_sha256=file_hash(Path(__file__).with_name("sac_branch_reconstruction.py")),
                  critic_factory_sha256=critic_factory_hash(),
                  critic_action_coordinates=arm.get("critic_action_coordinates", "raw"),
                  diagnostic_sha256=file_hash(args.diagnostic / "report.json"),
                  model_sha256=file_hash(weights), replay_sha256=model_check["replay_sha256"],
                  states_source_sha256=file_hash(args.states_from / "report.json"),
                  states_replay_sha256=storage_check["replay_sha256"],
                  collection_audit_adapter_sha256=file_hash(Path(__file__).with_name("sac_stratified_diagnostic_audit.py")),
                  selected_trace_indices=selected, samples=args.samples, cases=[],
                  missing_providers=sorted({"fixed_keep", "fixed_follow"} - set(selected)),
                  note="First actual terminal branch per provider, selected before Q. Same parent state re-executed using recorded main actions. Mean and stochastic actions use the actual Policy inference path. Q-minus-reward is a terminal error ONLY when this alternative action actually terminates. No replay or weight writes.")
    write_report(args.output, report)
    try:
        with jax.default_device(gpu):
            state = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                               actor=diagnostic_actor(arm, restoring=True),
                               critic=diagnostic_critic(arm, restoring=True))
            state = jax.device_put(serialization.from_bytes(state, weights.read_bytes()), gpu)
            assert int(state.updates) == source["updates"] and int(state.bc_updates) == 0
            mean_policy = Policy(state.actor, deterministic=True)
            sample_policy = Policy(state.actor, seed=77119)
            @jax.jit
            def qvalue(obs, raw):
                return jnp.minimum(state.q1.apply_fn({"params": state.q1.params}, obs, raw),
                                   state.q2.apply_fn({"params": state.q2.params}, obs, raw))
        scenes = {(r["scene_id"], r["provider"]): path for r, path in selected_scenes(args.manifest, "train")}
        for provider, trace_ix in selected.items():
            row = rows[trace_ix]
            ix = 5000 + trace_ix
            parent_rows = [(i, rows[i]) for i in branch_history_indices(rows, trace_ix)]
            assert parent_rows[0][1]["pre_tick"] == 0
            with jax.default_device(cpu):
                env = SACPlanningEnvironment()
                boundary = env.reset(parse_scenario(scenes[(row["scene_id"], provider)].read_bytes()),
                                     seed=origin["seed"], provider_spec=provider)
                for parent_i, parent in parent_rows:
                    assert env.tick == parent["pre_tick"]
                    pi = 5000 + parent_i
                    packed = pack_observation(boundary.sac_observation)
                    np.testing.assert_allclose(packed[0], a["obs"][pi], atol=1e-6, rtol=0)
                    np.testing.assert_array_equal(packed[1], a["fields"][pi])
                    np.testing.assert_array_equal(packed[2], a["items"][pi])
                    if parent_i == trace_ix:
                        break
                    step = collect_step(env, boundary,
                                        lambda e, t: proposal_from_action(e, t, a["raw_u"][pi]))
                    np.testing.assert_allclose(step.reward.training, a["reward"][pi], atol=1e-8, rtol=0)
                    assert not (step.result.terminated or step.result.truncated)
                    boundary = step.boundary
            with jax.default_device(gpu):
                obs = tensor_observation(boundary.sac_observation)
                actions = [("stored_branch", a["raw_u"][ix]),
                           ("current_mean", mean_policy.action(boundary.sac_observation))]
                actions += [(f"sample{k}", sample_policy.action(boundary.sac_observation)) for k in range(args.samples)]
                q_values = [float(np.asarray(qvalue(obs, jax.device_put(raw[None], gpu)))[0]) for _, raw in actions]
            case = dict(scene_id=row["scene_id"], provider=provider, trace_index=trace_ix,
                        replay_index=ix, tick=env.tick, parent_episode_uid=row["episode_uid"], actions={})
            report["cases"].append(case)
            with jax.default_device(cpu):
                before = observation_fingerprint(env._boundary().sac_observation)
                for (name, raw), q in zip(actions, q_values):
                    branch = fork_environment(env)
                    step = collect_step(branch, boundary,
                                        lambda e, t: proposal_from_action(e, t, raw))
                    assert step.transition is not None
                    value = dict(q=q, reward=step.reward.training, bootstrap=step.transition.bootstrap,
                                 outcome=step.result.terminal_reason.value if step.result.terminal_reason else None,
                                 check_valid=step.result.check_result.valid, check_reason=step.result.check_result.reason,
                                 source=step.result.source.value, raw_u=np.asarray(raw).tolist(),
                                 next_observation_sha256=observation_fingerprint(step.boundary.sac_observation),
                                 terminal_q_error=None if step.transition.bootstrap else q-step.reward.training)
                    case["actions"][name] = value
                    if name == "stored_branch":
                        np.testing.assert_allclose(step.reward.training, a["reward"][ix], atol=1e-8, rtol=0)
                        packed = pack_observation(step.boundary.sac_observation)
                        np.testing.assert_allclose(packed[0], a["next_obs"][ix], atol=1e-6, rtol=0)
                        np.testing.assert_array_equal(packed[1], a["next_fields"][ix])
                        np.testing.assert_array_equal(packed[2], a["next_items"][ix])
                        assert step.transition.bootstrap == a["bootstrap"][ix]
                    assert observation_fingerprint(env._boundary().sac_observation) == before
                    write_report(args.output, report)
            print(json.dumps({k: v for k, v in case.items() if k != "actions"}), flush=True)
        assert versions(args.manifest) == version and file_hash(weights) == report["model_sha256"]
        assert file_hash(args.diagnostic / "replay.npz") == report["replay_sha256"]
        assert file_hash(args.states_from / "replay.npz") == report["states_replay_sha256"]
        assert file_hash(args.states_from / "report.json") == report["states_source_sha256"]
        assert serialization.to_bytes(state) == weights.read_bytes()
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output, report)


if __name__ == "__main__":
    main()
