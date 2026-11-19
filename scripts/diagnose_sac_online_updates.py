#!/usr/bin/env python3
"""Fresh short online diagnostic with inherited self warmup, not a production run."""

import argparse
import json
import math
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
    for name in ("checkpoint", "configuration", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--continue-from", type=Path,
                        help="Completed same-configuration diagnostic; --updates is the total target")
    parser.add_argument("--collection-seed", type=int,
                        help="Explicit scene/policy RNG restart seed for continuation only")
    parser.add_argument("--failure-branch-samples", type=int, default=0,
                        help="Extra real current-policy steps per recent train failure state")
    parser.add_argument("--failure-branch-window", type=int, default=6)
    parser.add_argument("--failure-branch-selection", choices=("recent", "moving_rejections", "moving_predecessors"), default="recent")

    parser.add_argument("--critic-action-gain", type=float,
                        help="Opt-in positive critic input scale; no action clipping or loss change")
    parser.add_argument("--critic-action-coordinates", choices=("raw", "whitened", "physical_whitened"),
                        help="Explicit critic feature coordinates; executed/stored raw u is unchanged")
    parser.add_argument(
        "--zero-critic-action-init",
        action="store_true",
        help="Separate initialization trial: zero action rows of both critics' first layer",
    )
    parser.add_argument(
        "--mean-gain",
        type=float,
        help="Explicit diagnostic override of the approved linear mean gain",
    )
    parser.add_argument(
        "--save-replay",
        action="store_true",
        help="Persist the actual replay for read-only coverage audits",
    )
    args = parser.parse_args()
    if not 1 <= args.updates <= (5000 if args.continue_from else 1000):
        parser.error("fresh budget must be 1..1000; continuation total at most5000")
    if args.collection_seed is not None and (not args.continue_from or args.collection_seed < 0):
        parser.error("collection seed requires continuation and must be nonnegative")
    if args.mean_gain is not None and (
        not math.isfinite(args.mean_gain) or args.mean_gain <= 0
    ):
        parser.error("mean gain must be finite and positive")
    if args.critic_action_gain is not None and (
        not math.isfinite(args.critic_action_gain) or args.critic_action_gain <= 0
    ):
        parser.error("critic action gain must be finite and positive")
    from scripts.sac_failure_branches import FailureBranches
    try:
        FailureBranches(args.failure_branch_samples, args.failure_branch_window, args.failure_branch_selection)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import numpy as np
    from flax import serialization
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.networks import Actor, Critic
    from src.training.sac.learner import initialize, update
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import Replay
    from src.training.sac.collector import collect_step
    from src.contracts.observation import CoreObservation
    from src.contracts.planning import to_wire
    from src.scenarios.womd20 import parse_scenario
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from scripts.sac_experiment_runtime import model_proposer
    from scripts.sac_diagnostic_critic import diagnostic_critic, critic_factory_hash

    config = json.loads((args.configuration / "report.json").read_text())
    variant = dict(config["arms"][args.variant])
    overrides = {} if args.mean_gain is None else dict(mean_gain=args.mean_gain)
    if args.critic_action_gain is not None:
        overrides["critic_action_gain"] = args.critic_action_gain
    if args.critic_action_coordinates is not None:
        overrides["critic_action_coordinates"] = args.critic_action_coordinates
    if args.zero_critic_action_init:
        overrides["critic_action_kernel_init"] = "zero"
    variant.update(overrides)
    version = versions(args.manifest)
    if config["versions"] != version or variant["status"] != "COMPLETE":
        raise ValueError("requires a completed matching-runtime configuration")
    seed = config["seed"]
    meta = json.loads((args.checkpoint / "manifest.json").read_text())
    origin = meta["run_state"]["report"]
    if not (
        meta["run_state"]["seed"] == seed
        and origin["training_mode"] == "pure-sac"
        and origin["warmup_transitions"] == 5000
        and origin["bc_checkpoint_sha256"] is None
        and origin["initial_replay_size"] == 0
    ):
        raise ValueError("requires same-seed pure self warmup")
    if (
        meta["versions"]["manifest_sha256"] != version["manifest_sha256"]
        or meta["versions"]["contract_sha256"] != version["contract_sha256"]
    ):
        raise ValueError("warmup data/contract mismatch")
    if file_hash(args.checkpoint / "replay.npz") != meta["hashes"]["replay.npz"]:
        raise ValueError("replay checksum mismatch")
    replay = Replay.load(args.checkpoint / "replay.npz", meta["versions"])
    if (
        not 5000 <= replay.size < replay.capacity
        or replay.cursor != replay.size
        or np.any(replay.arrays["teacher_accepted"][:5000])
    ):
        raise ValueError("warmup prefix unavailable or contains teacher data")
    replay.size = replay.cursor = 5000
    replay.rng = np.random.default_rng(seed)
    # Explicit mixed-lineage diagnostic; source version is retained below.
    replay.versions = version
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    with jax.default_device(gpu):
        state = initialize(
            tensor_observation(CoreObservation.empty()),
            seed,
            actor=Actor(
                mean_gain=variant["mean_gain"],
                mean_basis=variant["mean_basis"],
                head_layer_norm=variant["actor_head_layer_norm"],
                mean_coordinate_scale=variant.get("mean_coordinate_scale", "raw"),
            ),
            critic=diagnostic_critic(variant),
        )
        init_mode = variant.get("critic_action_kernel_init", "standard")
        initialization_hash = None
        if init_mode == "zero":
            from scripts.sac_critic_initialization import (
                zero_critic_action_initialization,
            )

            state = zero_critic_action_initialization(state)
            initialization_hash = file_hash(
                Path(__file__).with_name("sac_critic_initialization.py")
            )
        elif init_mode != "standard":
            raise ValueError("unknown critic action initialization")
    report = dict(
        status="RUNNING",
        scope="SHORT_ONLINE_SAC_DIAGNOSTIC",
        qualified=False,
        versions=version,
        source_versions=meta["versions"],
        seed=seed,
        source_checkpoint_sha256=file_hash(args.checkpoint / "manifest.json"),
        configuration_sha256=file_hash(args.configuration / "report.json"),
        configuration_overrides=overrides,
        script_sha256=file_hash(__file__),
        critic_action_kernel_init=init_mode,
        initialization_script_sha256=initialization_hash,
        critic_factory_sha256=critic_factory_hash(),
        inherited_self_warmup=5000,
        new_transitions=0,
        updates=0,
        budget=args.updates,
        note="Fresh learner; first 5000 old own-policy warmup only. Scene/policy RNG restart: a new diagnostic branch, not exact continuation. Each new transition is inserted before one update. Validation never enters replay. Mid-episode budget endings are not successful episodes.",
        collection=dict(
            mode="actual_failure_branches" if args.failure_branch_samples else "sequential",
            samples=args.failure_branch_samples, window=args.failure_branch_window,
            selection=args.failure_branch_selection,
            main_transitions=0, branch_transitions=0, branch_terminal_transitions=0,
            branch_script_sha256=file_hash(Path(__file__).with_name("sac_failure_branches.py")),
            fork_script_sha256=file_hash(Path(__file__).with_name("diagnose_sac_q_values.py")),
            note="Uniform replay. Branches retain parent episode UID/tick; collection_id identifies independent actual steps. Branches are not full evaluation episodes. Nonterminal branch steps retain bootstrap; no artificial truncation.",
        ),
        episodes=[],
        arms={},
    )
    collection_seed = seed
    inherited_updates = 0
    if args.continue_from:
        from scripts.sac_diagnostic_continuation import validate_source, continuation_lineage
        expected_arm = dict(
            mean_gain=variant["mean_gain"], mean_basis=variant["mean_basis"],
            mean_coordinate_scale=variant.get("mean_coordinate_scale", "raw"),
            actor_head_layer_norm=variant["actor_head_layer_norm"], zero_q=variant["zero_q"],
            critic_action_coordinates=variant.get("critic_action_coordinates", "raw"),
            critic_action_kernel_init=init_mode,
            critic_action_gain=variant.get("critic_action_gain", 1.0),
            critic_factory_sha256=critic_factory_hash(),
        )
        source = validate_source(args.continue_from, version, expected_arm, report["collection"], args.updates)
        if (source["seed"] != seed
                or source["source_checkpoint_sha256"] != report["source_checkpoint_sha256"]
                or source["configuration_sha256"] != report["configuration_sha256"]):
            raise ValueError("continuation origin mismatch")
        # Audit actual train-only lineage before trusting the inherited data.
        from scripts.audit_sac_branch_replay import audit
        audit(args.continue_from, args.manifest)
        del replay
        replay = Replay.load(args.continue_from / "replay.npz", version)
        source_weights = (args.continue_from / "online-learner.msgpack").read_bytes()
        with jax.default_device(gpu):
            state = serialization.from_bytes(state, source_weights)
            state = jax.device_put(state, gpu)
        if serialization.to_bytes(state) != source_weights:
            raise ValueError("restored learner/optimizer/target/RNG serialization differs")
        inherited_updates = source["updates"]
        if (int(state.updates) != inherited_updates or int(state.bc_updates) != 0
                or replay.size != replay.cursor or replay.size != 5000 + inherited_updates
                or np.any(replay.arrays["teacher_accepted"][:replay.size])):
            raise ValueError("invalid restored learner/replay counters or teacher data")
        collection_seed = args.collection_seed if args.collection_seed is not None else seed + inherited_updates
        report.update(
            scope="BOUNDED_ONLINE_SAC_CONTINUATION", new_transitions=inherited_updates,
            updates=inherited_updates, collection=source["collection"], episodes=source["episodes"],
            continuation_history=source.get("continuation_history", []) + [
                continuation_lineage(args.continue_from, source, collection_seed)],
            continuation_script_sha256=file_hash(Path(__file__).with_name("sac_diagnostic_continuation.py")),
            restored_learner_bytes_identical=True,
            note="Same learner and replay continue; scene/policy RNG explicitly restart. "
                 "Cumulative histories include inherited own online experience. No validation/teacher/BC. "
                 "Previous budget boundary is not an artificial terminal or success.",
        )
    report["inherited_online_updates"] = inherited_updates
    report["collected_this_fork"] = 0
    report["collection_seed"] = collection_seed
    policy, proposer = model_proposer(state.actor, collection_seed, False, gpu)
    args.output.mkdir(parents=True)
    if args.continue_from:
        import shutil
        for filename in ("ticks.jsonl", "updates.jsonl"):
            shutil.copyfile(args.continue_from / filename, args.output / filename)
    candidates = selected_scenes(args.manifest, "train")
    rng = np.random.default_rng(collection_seed)
    def train_actual_step(environment, current_boundary, lineage):
        nonlocal state
        policy.actor = state.actor
        before_updates = int(state.updates)
        with jax.default_device(cpu):
            step = collect_step(environment, current_boundary, proposer, replay)
        result = step.result
        if step.transition is None:
            raise RuntimeError("invalid diagnostic transition; stop")
        with jax.default_device(gpu):
            state, metrics = update(state, replay.sample(256), replay.size)
        kind = lineage["kind"]
        report["collection"]["branch_transitions" if kind == "failure_branch" else "main_transitions"] += 1
        if kind == "failure_branch":
            report["collection"]["branch_terminal_transitions"] += int(result.terminated or result.truncated)
        report["new_transitions"] += 1
        report["collected_this_fork"] += 1
        report["updates"] = int(state.updates)
        with (args.output / "ticks.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        update=int(state.updates),
                        scene_id=row["scene_id"],
                        provider=row["provider"],
                        tick=environment.tick,
                        episode_uid=result.pre_token.episode_uid,
                        pre_tick=result.pre_token.sim_tick,
                        collection_id=report["new_transitions"],
                        policy_updates_before=before_updates,
                        lineage=lineage,
                        raw_proposal=to_wire(result.raw_proposal),
                        check_valid=result.check_result.valid,
                        check_reason=result.check_result.reason,
                        violations=list(result.check_result.violations),
                        source=result.source.value,
                        actual_control=to_wire(result.actual_control),
                        reward=step.reward.training,
                        bootstrap=step.transition.bootstrap,
                        delta_ticks=step.transition.delta_ticks,
                        terminal_reason=result.terminal_reason.value if result.terminal_reason else None,
                        terminated=result.terminated,
                        truncated=result.truncated,
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
        with (args.output / "updates.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        update=int(state.updates),
                        **{k: float(v) for k, v in metrics.items()},
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
        return step

    try:
        while int(state.updates) < args.updates:
            for index in rng.permutation(len(candidates)):
                row, path = candidates[index]
                with jax.default_device(cpu):
                    env = SACPlanningEnvironment()
                    boundary = env.reset(
                        parse_scenario(path.read_bytes()),
                        seed=seed,
                        provider_spec=row["provider"],
                    )
                episode = dict(
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    ticks=0,
                    accepted=0,
                    return_training=0.0,
                    outcome="IN_PROGRESS",
                    reasons={},
                )
                report["episodes"].append(episode)
                branches = FailureBranches(args.failure_branch_samples, args.failure_branch_window, args.failure_branch_selection)
                while int(state.updates) < args.updates:
                    with jax.default_device(cpu):
                        branches.remember(env, boundary)
                    step = train_actual_step(env, boundary, dict(kind="main"))
                    boundary = step.boundary
                    result = step.result
                    branches.after_step(result)
                    episode["ticks"] += 1
                    episode["accepted"] += int(result.check_result.valid)
                    episode["return_training"] += step.reward.training
                    reason = result.check_result.reason
                    episode["reasons"][reason] = episode["reasons"].get(reason, 0) + 1
                    if result.terminated or result.truncated:
                        episode["outcome"] = result.terminal_reason.value
                    write_report(args.output / "report.json", report)
                    if result.terminated or result.truncated:
                        # The main trajectory is finished. Branches are extra actual
                        # off-policy train experience, never invented full episodes.
                        with jax.default_device(cpu):
                            parent_token = env.token
                            from scripts.diagnose_sac_q_values import observation_fingerprint
                            parent_observation = observation_fingerprint(env._boundary().sac_observation)
                            for branch_env, branch_boundary, lineage in branches.generate(result):
                                if int(state.updates) >= args.updates:
                                    break
                                train_actual_step(branch_env, branch_boundary, lineage)
                                write_report(args.output / "report.json", report)
                            if env.token != parent_token or observation_fingerprint(env._boundary().sac_observation) != parent_observation:
                                raise RuntimeError("branch changed parent environment")
                        break
                if episode["outcome"] == "IN_PROGRESS":
                    episode["outcome"] = "DIAGNOSTIC_BUDGET"
                print(json.dumps(episode), flush=True)
                if int(state.updates) >= args.updates:
                    break
        if versions(args.manifest) != version:
            raise RuntimeError("runtime changed during online diagnostic")
        weights = args.output / "online-learner.msgpack"
        weights.write_bytes(serialization.to_bytes(state))
        report["arms"]["online"] = dict(
            status="COMPLETE",
            mean_gain=variant["mean_gain"],
            mean_basis=variant["mean_basis"],
            mean_coordinate_scale=variant.get("mean_coordinate_scale", "raw"),
            actor_head_layer_norm=variant["actor_head_layer_norm"],
            zero_q=variant["zero_q"],
            critic_action_coordinates=variant.get("critic_action_coordinates", "raw"),
            critic_action_kernel_init=init_mode,
            critic_action_gain=variant.get("critic_action_gain", 1.0),
            critic_factory_sha256=critic_factory_hash(),
            updates=int(state.updates),
            learner_sha256=file_hash(weights),
        )
        if args.save_replay:
            recorded = [
                json.loads(line)["raw_proposal"]["raw_u"]
                for line in (args.output / "ticks.jsonl").read_text().splitlines()
            ]
            np.testing.assert_array_equal(
                replay.arrays["raw_u"][5000 : replay.size],
                np.asarray(recorded, np.float64),
            )
            if replay.size != 5000 + report["new_transitions"] or np.any(
                replay.arrays["teacher_accepted"][: replay.size]
            ):
                raise ValueError("unexpected diagnostic replay composition")
            replay.save(args.output / "replay.npz")
            report["saved_replay"] = dict(
                file="replay.npz",
                sha256=file_hash(args.output / "replay.npz"),
                size=replay.size,
                own_warmup=5000,
                new_online=report["new_transitions"],
            )
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
