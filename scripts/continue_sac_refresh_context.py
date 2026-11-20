#!/usr/bin/env python3
"""Continue a fresh refresh experiment after rebuilding its actual live context.

Existing fresh runners and their results stay unchanged. Every added record is a
new actual policy/environment step followed by one original full SAC update.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCOPE = "OWN_REFRESH_CONTEXT_CONTINUATION"
SOURCE_FILES = ("report.json", "online-learner.msgpack", "replay.npz", "ticks.jsonl",
                "updates.jsonl", "initial-learner.msgpack", "initial-state-audit.json")


def restore_trained_learner(template, weights, device):
    """Rebuild JAX's trained mapping order without converting scalar leaves."""
    import jax
    import numpy as np
    from flax import serialization

    state = serialization.from_bytes(template, weights)
    # Flax uses the fresh template's insertion order when restoring dictionaries.
    # Saved trained states have JAX pytree order after JIT. An identity traversal
    # restores that ordering while leaving all arrays and Python scalars intact.
    # Place array leaves for inference, but preserve Python counters as scalars.
    # Whole-state device_put would silently convert those counters to arrays.
    state = jax.tree_util.tree_map(
        lambda leaf: jax.device_put(leaf, device)
        if isinstance(leaf, (np.ndarray, jax.Array)) else leaf, state)
    if serialization.to_bytes(state) != weights:
        raise ValueError("restored learner/optimizer/target/RNG bytes differ")
    return state


def validate_budget(source, total_budget):
    inherited = source.get("updates")
    if (type(inherited) is not int or type(total_budget) is not int
            or not 0 < inherited < total_budget <= 10000
            or source.get("status") != "COMPLETE"
            or source.get("inherited_online_updates") != 0
            or source.get("continuation_history")
            or source.get("inherited_self_warmup") != 5000
            or source.get("new_transitions") != inherited):
        raise ValueError("requires a completed fresh source and a larger total budget <=10000")
    allowed = {
        "OWN_EARLY_MOVING_REFRESH_SAC_COMPARISON": "early_moving_refresh",
        "OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON": "provider_protected_refresh",
    }
    key = allowed.get(source.get("scope"))
    if key is None or key not in source or any(k in source for k in allowed.values() if k != key):
        raise ValueError("requires one unambiguous supported refresh contract")
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "manifest", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, help="Total, including inherited updates")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads((args.source/"report.json").read_text())
    contract = validate_budget(source, args.updates)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frozen = {str(args.source/n): sha(args.source/n) for n in SOURCE_FILES}
    frozen[str(Path(__file__))] = sha(__file__)
    frozen[str(Path(__file__).with_name("sac_refresh_context_restore.py"))] = sha(
        Path(__file__).with_name("sac_refresh_context_restore.py"))
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import jax
    import numpy as np
    from flax import serialization
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from scripts.sac_experiment_runtime import model_proposer
    from scripts.sac_refresh_context_restore import restore_context
    from scripts.sac_terminal_retention_sampling import sample_recent
    from scripts.sac_failure_branches import FailureBranches
    from scripts.sac_failure_complete_path import collect_complete_failure_path
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from src.contracts.observation import CoreObservation
    from src.contracts.planning import to_wire
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.training.sac.collector import collect_step
    from src.training.sac.learner import initialize, update
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import Replay
    from src.training.sac.runtime import selected_scenes, versions, write_report

    version = versions(args.manifest)
    if version != source["versions"]:
        raise ValueError("source runtime/data changed")
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    inherited = source["updates"]
    report = copy.deepcopy(source)
    report.update(status="RESTORING", scope=SCOPE, qualified=False, budget=args.updates,
        inherited_online_updates=inherited, collected_this_fork=0,
        script_sha256=sha(__file__), source_script_sha256=source["script_sha256"],
        continuation_history=[dict(source_directory=str(args.source.resolve()),
            source_hashes={n: frozen[str(args.source/n)] for n in SOURCE_FILES},
            inherited_online_updates=inherited, source_scope=source["scope"],
            method="replay_actual_context_without_collection_rng_restart")],
        restored_learner_bytes_identical=False,
        note="Actual collector context and policy/scene RNG are reconstructed; learner/replay RNG continue. "
             "No reset at inherited budget, no relabeling, no teacher/BC or loss/reward/checker/PD change.")
    args.output.mkdir(parents=True)
    write_report(args.output/"report.json", report)
    try:
        def progress(done, total):
            write_report(args.output/"restoration-progress.json", dict(status="RUNNING", restored=done, total=total))
        with jax.default_device(cpu):
            context = restore_context(args.source, args.checkpoint, args.manifest, progress=progress)
        write_report(args.output/"restored-context.json", context.evidence)
        arm = source["arms"]["online"]
        weights = (args.source/"online-learner.msgpack").read_bytes()
        with jax.default_device(gpu):
            state = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                actor=diagnostic_actor(arm, restoring=True), critic=diagnostic_critic(arm, restoring=True))
            state = restore_trained_learner(state, weights, gpu)
        if int(state.updates) != inherited or int(state.bc_updates) != 0:
            raise ValueError("restored learner counter/BC mismatch")
        replay = Replay.load(args.source/"replay.npz", version)
        if (replay.size != replay.cursor or replay.size != 5000+inherited
                or 5000+args.updates >= replay.capacity
                or np.any(replay.arrays["teacher_accepted"][:replay.size])):
            raise ValueError("invalid own replay or capacity")
        for name in ("ticks.jsonl", "updates.jsonl", "initial-learner.msgpack", "initial-state-audit.json"):
            shutil.copyfile(args.source/name, args.output/name)
        write_report(args.output/"restored-state-audit.json", dict(status="COMPLETE",
            source_learner_sha256=sha(args.source/"online-learner.msgpack"),
            restored_learner_sha256=hashlib.sha256(serialization.to_bytes(state)).hexdigest(),
            bytes_equal=True, updates=int(state.updates), bc_updates=int(state.bc_updates),
            actor_steps=int(state.actor.step), q1_steps=int(state.q1.step), q2_steps=int(state.q2.step),
            replay_size=replay.size, replay_rng=replay.rng.bit_generator.state,
            policy_rng=np.asarray(context.policy_rng).tolist(),
            context_sha256=sha(args.output/"restored-context.json")))
        report["restored_learner_bytes_identical"] = True
        report["status"] = "RUNNING"
        report["arms"]["online"]["status"] = "RUNNING"
        policy, proposer = model_proposer(state.actor, source["collection_seed"], False, gpu)
        with jax.default_device(gpu):
            policy.rng = jax.device_put(context.policy_rng, gpu)
        bank = context.refresh_bank
        env, boundary, branches = context.environment, context.boundary, context.failure_branches
        scene_rng, order, position = context.scene_rng, context.scene_permutation, context.scene_position
        candidates = selected_scenes(args.manifest, "train")
        row, path = candidates[order[position]]
        episode = report["episodes"][-1]
        episode["outcome"] = "IN_PROGRESS"

        def train_actual(environment, before, lineage):
            nonlocal state
            policy.actor = state.actor
            before_updates = int(state.updates)
            with jax.default_device(cpu):
                snapshot = fork_environment(environment) if lineage["kind"] != "successor_refresh" else None
                actual = collect_step(environment, before, proposer, replay)
            if actual.transition is None:
                raise RuntimeError("actual transition is invalid; no substitute experience")
            batch, digest = sample_recent(replay)
            with jax.default_device(gpu):
                state, metrics = update(state, batch, replay.size)
            if int(state.updates) != before_updates+1 or int(state.bc_updates) != 0:
                raise RuntimeError("one actual transition must have one full SAC update")
            kind, result = lineage["kind"], actual.result
            if kind == "successor_refresh":
                report["refresh_transitions"] += 1
                report["refresh_terminal_transitions"] += int(result.terminated or result.truncated)
            else:
                report["collection"]["main_transitions" if kind == "main" else "branch_transitions"] += 1
            if kind == "failure_branch":
                report["collection"]["branch_terminal_transitions"] += int(result.terminated or result.truncated)
            report["new_transitions"] += 1
            report["collected_this_fork"] += 1
            report["updates"] = int(state.updates)
            tick = dict(update=int(state.updates),
                scene_id=lineage["scene_id"] if kind == "successor_refresh" else row["scene_id"],
                provider=lineage["provider"] if kind == "successor_refresh" else row["provider"],
                tick=environment.tick, episode_uid=result.pre_token.episode_uid,
                pre_tick=result.pre_token.sim_tick, collection_id=report["new_transitions"],
                policy_updates_before=before_updates, lineage=lineage,
                raw_proposal=to_wire(result.raw_proposal), check_valid=result.check_result.valid,
                check_reason=result.check_result.reason, violations=list(result.check_result.violations),
                source=result.source.value, actual_control=to_wire(result.actual_control),
                reward=actual.reward.training, bootstrap=actual.transition.bootstrap,
                delta_ticks=actual.transition.delta_ticks,
                terminal_reason=result.terminal_reason.value if result.terminal_reason else None,
                terminated=result.terminated, truncated=result.truncated)
            with (args.output/"ticks.jsonl").open("a") as stream:
                stream.write(json.dumps(tick, allow_nan=False)+"\n")
            with (args.output/"updates.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(update=int(state.updates),
                    **{k: float(v) for k,v in metrics.items()}, replay_indices_sha256=digest,
                    replay_size=replay.size), allow_nan=False)+"\n")
            if kind != "successor_refresh":
                bank.after_base(snapshot, before, actual, collection_id=report["new_transitions"],
                    scene_id=row["scene_id"], provider=row["provider"], kind=kind)
            return actual

        def train_and_refresh(environment, before, lineage):
            actual = train_actual(environment, before, lineage)
            actual_id = report["new_transitions"]
            if int(state.updates) < args.updates:
                with jax.default_device(cpu):
                    draw = bank.draw()
                if draw is not None:
                    refresh_env, refresh_boundary, refresh_lineage = draw
                    refreshed = train_actual(refresh_env, refresh_boundary, refresh_lineage)
                    bank.after_refresh(refreshed, collection_id=report["new_transitions"])
                    bank.verify_preserved()
            return actual, actual_id

        while int(state.updates) < args.updates:
            with jax.default_device(cpu):
                branches.remember(env, boundary)
            actual, _ = train_and_refresh(env, boundary, dict(kind="main"))
            boundary, result = actual.boundary, actual.result
            branches.after_step(result)
            episode["ticks"] += 1
            episode["accepted"] += int(result.check_result.valid)
            episode["return_training"] += actual.reward.training
            reason = result.check_result.reason
            episode["reasons"][reason] = episode["reasons"].get(reason, 0)+1
            if result.terminated or result.truncated:
                episode["outcome"] = result.terminal_reason.value
            write_report(args.output/"report.json", report)
            if result.terminated or result.truncated:
                with jax.default_device(cpu):
                    parent_token = env.token
                    parent_obs = observation_fingerprint(env._boundary().sac_observation)
                    def branch_step(e, b, lineage):
                        value = train_and_refresh(e, b, lineage)
                        write_report(args.output/"report.json", report)
                        return value
                    collect_complete_failure_path(branches, result, horizon=200,
                        train_step=branch_step, can_step=lambda: int(state.updates) < args.updates)
                    if env.token != parent_token or observation_fingerprint(env._boundary().sac_observation) != parent_obs:
                        raise RuntimeError("actual branch changed its parent environment")
                if int(state.updates) >= args.updates:
                    break
                position += 1
                if position == len(order):
                    order, position = scene_rng.permutation(len(candidates)), 0
                row, path = candidates[order[position]]
                with jax.default_device(cpu):
                    env = SACPlanningEnvironment()
                    boundary = env.reset(parse_scenario(path.read_bytes()), seed=source["seed"], provider_spec=row["provider"])
                branches = FailureBranches(1, 6, "moving_predecessors")
                episode = dict(scene_id=row["scene_id"], provider=row["provider"], ticks=0,
                    accepted=0, return_training=0., outcome="IN_PROGRESS", reasons={})
                report["episodes"].append(episode)
        if episode["outcome"] == "IN_PROGRESS":
            episode["outcome"] = "DIAGNOSTIC_BUDGET"
        bank.verify_preserved()
        if versions(args.manifest) != version:
            raise RuntimeError("runtime changed during continuation")
        for p,h in frozen.items():
            if sha(p) != h:
                raise RuntimeError(f"source changed during continuation: {p}")
        if replay.size != 5000+int(state.updates) or np.any(replay.arrays["teacher_accepted"][:replay.size]):
            raise RuntimeError("invalid own-experience replay composition")
        weights_path = args.output/"online-learner.msgpack"
        weights_path.write_bytes(serialization.to_bytes(state))
        replay.save(args.output/"replay.npz")
        report["arms"]["online"].update(status="COMPLETE", updates=int(state.updates), learner_sha256=sha(weights_path))
        report["saved_replay"] = dict(file="replay.npz", sha256=sha(args.output/"replay.npz"),
            size=replay.size, own_warmup=5000, new_online=report["new_transitions"])
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output/"report.json", report)


if __name__ == "__main__":
    main()
