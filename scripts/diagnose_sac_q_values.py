#!/usr/bin/env python3
"""Frozen-policy same-state Q audit, with actual counterfactual Waymax steps.

No replay, learner update, threshold change, or policy selection in production.
"""

import argparse
import copy
from dataclasses import asdict
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


def fork_environment(environment):
    """Share immutable JAX snapshots/read-only geometry; isolate mutable ledgers.

    Diagnostic-only for the current KEEP/FOLLOW environment. The parent rollout
    is independently replayed against saved frames to detect branch interference.
    """
    branch = copy.copy(environment)
    branch.ledger = copy.deepcopy(environment.ledger)
    branch._step_cache = environment._step_cache.copy()
    return branch


def probe_ticks(frames):
    """Fixed cadence plus first rejection/braking/stop and pre-terminal state."""
    ticks = set(range(0, len(frames) - 1, 10)) | {len(frames) - 2}
    for predicate in (
        lambda f: not f["accepted"],
        lambda f: f["source"] in ("BRAKE", "EMERGENCY"),
        lambda f: f["ego"]["speed"] < 0.1,
    ):
        first = next((f for f in frames[1:] if predicate(f)), None)
        if first is not None:
            ticks.add(first["proposal_tick"])
    return sorted(t for t in ticks if t >= 0)


def observation_fingerprint(core):
    """All policy features/field masks/item masks, including failure feedback."""
    from src.training.sac.replay import pack_observation

    digest = hashlib.sha256()
    for array in pack_observation(core):
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "evaluation", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--continuations", type=int, default=2)
    parser.add_argument(
        "--terminal-only",
        action="store_true",
        help="Audit only the final pre-step state of each saved rollout",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.samples < 2 or args.continuations < 0:
        parser.error(
            "at least two candidate samples and nonnegative continuations required"
        )
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from src.contracts.observation import CoreObservation
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.collector import collect_step
    from src.training.sac.learner import initialize
    from src.training.sac.networks import (
        Actor,
        Critic,
        normalized_action,
        observation_speed,
        sample_action,
    )
    from src.training.sac.policy import proposal_from_action, tensor_observation
    from src.training.sac.runtime import selected_scenes, versions, write_report
    from src.training.sac.teacher import FrenetTeacher
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from scripts.sac_diagnostic_critic import diagnostic_critic

    source = json.loads((args.diagnostic / "report.json").read_text())
    evaluation = json.loads((args.evaluation / "report.json").read_text())
    version = versions(args.manifest)
    if any(
        d["status"] != "COMPLETE" or d["versions"] != version
        for d in (source, evaluation)
    ):
        raise ValueError("completed identical-runtime sources required")
    arm = source["arms"]["online"]
    weights = args.diagnostic / "online-learner.msgpack"
    weights_hash = file_hash(weights)
    if not (
        weights_hash == arm["learner_sha256"] == evaluation["learner_sha256"]
        and evaluation["diagnostic_sha256"]
        == file_hash(args.diagnostic / "report.json")
    ):
        raise ValueError("source provenance mismatch")
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    with jax.default_device(gpu):
        initial = initialize(
            tensor_observation(CoreObservation.empty()),
            source["seed"],
            actor=diagnostic_actor(arm, restoring=True),
            critic=diagnostic_critic(arm, restoring=True),
        )
        state = jax.device_put(
            serialization.from_bytes(initial, weights.read_bytes()), gpu
        )
        forward = jax.jit(state.actor.apply_fn)
        q1, q2 = jax.jit(state.q1.apply_fn), jax.jit(state.q2.apply_fn)
    if int(state.updates) != arm["updates"] or int(state.bc_updates) != 0:
        raise ValueError("unexpected learner counters")
    alpha = float(jnp.exp(state.log_alpha))

    def observation(boundary):
        return jax.device_put(tensor_observation(boundary.sac_observation), gpu)

    def mean_propose(e, token):
        with jax.default_device(gpu):
            mu, _ = forward({"params": state.actor.params}, observation(e._boundary()))
        return proposal_from_action(e, token, np.asarray(mu[0], np.float64))

    def target_value(boundary, key):
        with jax.default_device(gpu):
            obs = observation(boundary)
            obs = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, args.samples, axis=0), obs
            )
            mu, rho = forward({"params": state.actor.params}, obs)
            u, ell, _ = sample_action(mu, rho, observation_speed(obs), key)
            values = (
                jnp.minimum(
                    q1({"params": state.target1}, obs, u),
                    q2({"params": state.target2}, obs, u),
                )
                - alpha * ell
            )
        return float(values.mean()), float(values.std())

    def continuation(e, boundary, proposal, seed):
        """Full discounted soft return: fixed first action, stochastic policy suffix.

        No entropy term at the conditioned first action. Same RNG stream per
        candidate within a state/replicate; no bootstrap at any contract ending.
        """
        branch = fork_environment(e)
        token, original_state, memory = e.token, e._state, e.memory
        key = jax.random.PRNGKey(seed)
        value = plain = 0.0
        rejected = braking = stopped = 0
        for index in range(200):
            ell = 0.0
            if index == 0:
                action = proposal.raw_u
            else:
                key, draw = jax.random.split(key)
                with jax.default_device(gpu):
                    obs = observation(boundary)
                    mu, rho = forward({"params": state.actor.params}, obs)
                    u, logp, _ = sample_action(mu, rho, observation_speed(obs), draw)
                action, ell = np.asarray(u[0]), float(logp[0])
            with jax.default_device(cpu):
                step = collect_step(
                    branch,
                    boundary,
                    lambda env, tok: proposal_from_action(env, tok, action),
                )
            if step.transition is None:
                raise ValueError("invalid counterfactual transition")
            boundary = step.boundary
            value += 0.99**index * (step.reward.training - alpha * ell)
            plain += 0.99**index * step.reward.training
            rejected += int(not step.result.check_result.valid)
            braking += int(step.result.source.value in ("BRAKE", "EMERGENCY"))
            stopped += int(branch.ego.speed < 0.1)
            if step.result.terminated or step.result.truncated:
                break
        else:
            raise ValueError("continuation did not reach contract terminal boundary")
        assert e.token == token and e._state is original_state and e.memory == memory
        return dict(
            seed=seed,
            soft_return=value,
            discounted_reward=plain,
            outcome=step.result.terminal_reason.value,
            ticks=index + 1,
            rejected=rejected,
            braking=braking,
            stationary=stopped,
        )

    report = dict(
        status="RUNNING",
        qualified=False,
        scope="FROZEN_Q_SAME_STATE_AUDIT",
        versions=version,
        seed=source["seed"],
        learner_sha256=weights_hash,
        diagnostic_sha256=file_hash(args.diagnostic / "report.json"),
        evaluation_sha256=file_hash(args.evaluation / "report.json"),
        script_sha256=file_hash(__file__),
        candidate_samples=args.samples,
        continuation_replicates=args.continuations,
        probe_selection=(
            "terminal_only" if args.terminal_only else "cadence_and_failure_events"
        ),
        alpha=alpha,
        episodes=[],
        note="No training/replay changes. Q predicts reward after safety fallback, not a safety probability. One-step target uses target critics and is not independent ground truth. Full continuations use frozen stochastic policy and entropy coordinates, no current-action entropy. Rule-based/zero candidates may be out of policy support. Selected validation states and small MC samples are diagnostic only.",
    )
    args.output.mkdir(parents=True)
    pairs = {tuple(pair) for pair in evaluation["selected_pairs"]}
    try:
        for row, path in selected_scenes(args.manifest, "validation"):
            if (row["scene_id"], row["provider"]) not in pairs:
                continue
            evidence_path = (
                args.evaluation / f"updated-{row['scene_id']}-{row['provider']}.json"
            )
            expected = json.loads(evidence_path.read_text())
            ticks = (
                [len(expected["frames"]) - 2]
                if args.terminal_only
                else probe_ticks(expected["frames"])
            )
            first_reject = next(
                (
                    f["proposal_tick"]
                    for f in expected["frames"][1:]
                    if not f["accepted"]
                ),
                0,
            )
            continuation_ticks = (
                {0, first_reject}
                if row["provider"] == "fixed_keep"
                and args.continuations
                and not args.terminal_only
                else set()
            )
            with jax.default_device(cpu):
                e = SACPlanningEnvironment()
                boundary = e.reset(
                    parse_scenario(path.read_bytes()),
                    seed=source["seed"],
                    provider_spec=row["provider"],
                )
            episode = dict(
                scene_id=row["scene_id"],
                provider=row["provider"],
                evidence_sha256=file_hash(evidence_path),
                probe_ticks=ticks,
                continuation_ticks=sorted(continuation_ticks),
                probes=[],
                status="RUNNING",
            )
            report["episodes"].append(episode)
            teacher = FrenetTeacher()
            base_seed = int(row["scene_id"][:8], 16) % (2**31)
            for expected_frame in expected["frames"][1:]:
                if e.tick in ticks:
                    with jax.default_device(gpu):
                        obs = observation(boundary)
                        mu, rho = forward({"params": state.actor.params}, obs)
                        speed = observation_speed(obs)
                        repeated = lambda x: jnp.repeat(x, args.samples, axis=0)
                        sampled, _, _ = sample_action(
                            repeated(mu),
                            repeated(rho),
                            repeated(speed),
                            jax.random.PRNGKey(base_seed + e.tick),
                        )
                    with jax.default_device(cpu):
                        candidates = {
                            "mean": proposal_from_action(
                                e, e.token, np.asarray(mu[0], np.float64)
                            ),
                            "zero": proposal_from_action(
                                e, e.token, np.zeros((30, 2), np.float64)
                            ),
                            "rule_based": teacher.propose(e, e.token).proposal,
                        }
                        candidates.update(
                            {
                                f"sample_{i}": proposal_from_action(
                                    e, e.token, np.asarray(u)
                                )
                                for i, u in enumerate(sampled)
                            }
                        )
                    names = list(candidates)
                    with jax.default_device(gpu):
                        actions = jnp.asarray(
                            np.stack([p.raw_u for p in candidates.values()])
                        )
                        batch = jax.tree_util.tree_map(
                            lambda x: jnp.repeat(x, len(names), axis=0), obs
                        )
                        qa = np.asarray(q1({"params": state.q1.params}, batch, actions))
                        qb = np.asarray(q2({"params": state.q2.params}, batch, actions))
                        sigma = 0.005 + 0.025 * jax.nn.sigmoid(rho.astype(jnp.float64))
                        epsilon = (
                            normalized_action(actions - mu, speed)
                            * (0.02 * np.sqrt(2 * np.pi * np.e))
                            / sigma[:, None, :]
                        )
                        distances = np.asarray(
                            jnp.sqrt(jnp.mean(epsilon**2, axis=(-2, -1)))
                        )
                    probe = dict(
                        tick=e.tick,
                        speed=e.ego.speed,
                        previous_source=expected["frames"][e.tick]["source"],
                        candidates={},
                        continuations={},
                    )
                    parent_state, parent_token, parent_memory = (
                        e._state,
                        e.token,
                        e.memory,
                    )
                    for index, name in enumerate(names):
                        branch = fork_environment(e)
                        with jax.default_device(cpu):
                            step = collect_step(
                                branch, boundary, lambda env, tok: candidates[name]
                            )
                        if step.transition is None:
                            raise ValueError("invalid one-step transition")
                        value, spread = (
                            target_value(
                                step.boundary,
                                jax.random.PRNGKey(base_seed + 10000 + e.tick),
                            )
                            if step.transition.bootstrap
                            else (0.0, 0.0)
                        )
                        target = (
                            step.reward.training
                            + 0.99**step.transition.delta_ticks * value
                        )
                        probe["candidates"][name] = dict(
                            q1=float(qa[index]),
                            q2=float(qb[index]),
                            min_q=float(min(qa[index], qb[index])),
                            policy_sigma_distance_rms=float(distances[index]),
                            raw_u=candidates[name].raw_u.tolist(),
                            check=asdict(step.result.check_result),
                            source=step.result.source.value,
                            reward=asdict(step.reward),
                            next_xy=branch.ego.position.tolist(),
                            next_velocity=branch.ego.velocity.tolist(),
                            actual_control=asdict(step.result.actual_control),
                            outcome=(
                                step.result.terminal_reason.value
                                if step.result.terminal_reason
                                else None
                            ),
                            bootstrap=bool(step.transition.bootstrap),
                            next_observation_sha256=observation_fingerprint(
                                step.boundary.sac_observation
                            ),
                            delta_ticks=step.transition.delta_ticks,
                            one_step_soft_target=target,
                            sampled_next_soft_value_std=spread,
                            q_minus_target=float(min(qa[index], qb[index]) - target),
                        )
                        assert (
                            e._state is parent_state
                            and e.token == parent_token
                            and e.memory == parent_memory
                        )
                    if e.tick in continuation_ticks:
                        on_policy = [
                            n for n in names if n not in ("zero", "rule_based")
                        ]
                        top = max(
                            on_policy, key=lambda n: probe["candidates"][n]["min_q"]
                        )
                        valid = [
                            n for n in names if probe["candidates"][n]["check"]["valid"]
                        ]
                        chosen = ["mean", top]
                        if valid:
                            chosen.append(
                                max(
                                    valid, key=lambda n: probe["candidates"][n]["min_q"]
                                )
                            )
                        probe["continuation_selection"] = dict(
                            top_on_policy=top, valid_available=bool(valid)
                        )
                        for name in dict.fromkeys(chosen):
                            probe["continuations"][name] = [
                                continuation(
                                    e,
                                    boundary,
                                    candidates[name],
                                    base_seed + 20000 + e.tick * 10 + i,
                                )
                                for i in range(args.continuations)
                            ]
                    episode["probes"].append(probe)
                    write_report(args.output / "report.json", report)
                    print(
                        json.dumps(
                            dict(
                                scene=row["scene_id"],
                                provider=row["provider"],
                                tick=e.tick,
                                candidates=len(candidates),
                                continuations=list(probe["continuations"]),
                            )
                        ),
                        flush=True,
                    )
                with jax.default_device(cpu):
                    step = collect_step(e, boundary, mean_propose)
                boundary = step.boundary
                np.testing.assert_allclose(
                    e.ego.position, expected_frame["ego"]["xy"], atol=1e-6, rtol=0
                )
                np.testing.assert_allclose(
                    e.ego.velocity, expected_frame["ego"]["velocity"], atol=1e-6, rtol=0
                )
                assert step.result.source.value == expected_frame["source"]
                assert step.result.check_result.reason == expected_frame["reason"]
                assert (
                    step.result.terminal_reason.value
                    if step.result.terminal_reason
                    else None
                ) == expected_frame["outcome"]
                np.testing.assert_allclose(
                    step.reward.training, expected_frame["reward"], atol=1e-8, rtol=0
                )
            assert e.tick == expected["summary"]["ticks"]
            episode.update(
                status="COMPLETE",
                outcome=e.terminal_reason.value,
                reproduced_ticks=e.tick,
            )
            write_report(args.output / "report.json", report)
        if versions(args.manifest) != version or file_hash(weights) != weights_hash:
            raise RuntimeError("source changed during read-only audit")
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
