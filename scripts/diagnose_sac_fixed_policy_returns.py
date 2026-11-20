#!/usr/bin/env python3
"""Compare frozen-policy critic snapshots against actual sampled-path returns.

This evaluates training experience, not held-out driving or exact expected Q.
It never changes a policy, critic, replay, reward, or terminal condition.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.audit_sac_fixed_policy_critic import audit
    from scripts.sac_bellman_trace import decompose
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from src.contracts.observation import CoreObservation
    from src.training.sac.learner import initialize
    from src.training.sac.networks import observation_speed, sample_action
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import unpack_observation

    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    checked = audit(args.diagnostic, args.source)
    report = read(args.diagnostic / "report.json")
    source = read(args.source / "report.json")
    ticks = [json.loads(line) for line in (args.diagnostic / "ticks.jsonl").read_text().splitlines()]
    inherited = report["inherited_replay_size"]
    with np.load(args.diagnostic / "diagnostic-replay.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k][inherited:].copy() for k in
                  ("obs", "fields", "items", "raw_u", "reward", "bootstrap", "delta_ticks")}
    assert len(ticks) == len(arrays["raw_u"]) == report["actual_transitions"]
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        template = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                              actor=diagnostic_actor(source["arms"]["online"], restoring=True),
                              critic=diagnostic_critic(source["arms"]["online"], restoring=True))
        states = {}
        for name, path in (("before", args.source / "online-learner.msgpack"),
                           ("after", args.diagnostic / "critic-evaluation-state.msgpack")):
            state = jax.device_put(serialization.from_bytes(template, path.read_bytes()), gpu)
            assert serialization.to_bytes(state) == path.read_bytes()
            states[name] = state
        state = states["before"]
        assert serialization.to_bytes(state.actor) == serialization.to_bytes(states["after"].actor)
        alpha = float(jnp.exp(state.log_alpha))
        assert alpha == report["alpha"] == float(jnp.exp(states["after"].log_alpha))
        forward = jax.jit(state.actor.apply_fn)
        apply1, apply2 = state.q1.apply_fn, state.q2.apply_fn

        @jax.jit
        def values(params1, params2, observation, raw):
            return jnp.minimum(apply1({"params": params1}, observation, raw),
                               apply2({"params": params2}, observation, raw))

        rows = []
        groups = {}
        maximum_action_difference = 0.
        rng = None
        for i, tick in enumerate(ticks):
            if tick["branch_step"] == 0:
                rng = jax.random.PRNGKey(tick["policy_seed"])
            assert rng is not None
            observation = unpack_observation(*(jnp.asarray(arrays[k][i:i+1])
                                               for k in ("obs", "fields", "items")))
            mu, rho = forward({"params": state.actor.params}, observation)
            rng, key = jax.random.split(rng)
            sampled, ell, _ = sample_action(mu, rho, observation_speed(observation), key)
            raw = arrays["raw_u"][i]
            difference = float(np.max(np.abs(np.asarray(sampled[0]) - raw)))
            # Fixed before running: allow only tiny forward numerical variation.
            # Never infer epsilon by inverting a tiny physical covariance.
            assert difference <= 1e-9, (i, difference)
            maximum_action_difference = max(maximum_action_difference, difference)
            row = {k: tick[k] for k in ("collection_id", "rollout", "root_index", "policy_seed",
                                        "branch_step", "scene_id", "provider", "pre_tick", "bootstrap", "outcome")}
            row.update(reward=float(arrays["reward"][i]), ell=float(np.asarray(ell)[0]),
                       action_reproduction_max_difference=difference)
            for name, snapshot in states.items():
                def q(p1, p2):
                    return float(np.asarray(values(p1, p2, observation, jnp.asarray(raw[None])))[0])
                row[name] = dict(q=q(snapshot.q1.params, snapshot.q2.params),
                                 target_q=q(snapshot.target1, snapshot.target2))
            rows.append(row)
            groups.setdefault(tick["rollout"], []).append(row)
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(evaluated=i+1, total=len(ticks))), flush=True)

    episodes, all_complete = [], []
    excluded = []
    for rollout, group in groups.items():
        if group[-1]["bootstrap"]:
            excluded.append(dict(rollout=rollout, ticks=len(group), reason="DIAGNOSTIC_BUDGET"))
            continue
        assert all(r["bootstrap"] for r in group[:-1])
        entropy = [0.] + [r["ell"] for r in group[1:]]
        episode = dict(rollout=rollout, provider=group[0]["provider"], scene_id=group[0]["scene_id"],
                       ticks=len(group), outcome=group[-1]["outcome"],
                       undiscounted_return=sum(r["reward"] for r in group))
        for name in states:
            result = decompose([r["reward"] for r in group], entropy,
                               [r[name]["q"] for r in group], [r[name]["target_q"] for r in group],
                               [r["bootstrap"] for r in group], alpha)
            episode[name] = result["summary"]
            for row, accounted in zip(group, result["rows"]):
                row[name].update(accounted)
        assert episode["before"]["soft_return"] == episode["after"]["soft_return"]
        episodes.append(episode)
        all_complete.extend(group)
    assert episodes

    def summarize(selected):
        result = dict(ticks=len(selected))
        if selected:
            for name in states:
                error = np.asarray([r[name]["q_minus_observed_return"] for r in selected])
                result[name] = dict(mean_error=float(error.mean()), mae=float(np.abs(error).mean()),
                                    max_absolute_error=float(np.abs(error).max()))
        return result

    summaries = {}
    for provider in ("all", "fixed_keep", "fixed_follow"):
        selected = [r for r in all_complete if provider == "all" or r["provider"] == provider]
        summaries[provider] = dict(all_ticks=summarize(selected),
            nonterminal=summarize([r for r in selected if r["bootstrap"]]),
            terminal=summarize([r for r in selected if not r["bootstrap"]]),
            first_stochastic_action=summarize([r for r in selected if r["branch_step"] == 0]))
    for path, expected in checked["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    paths = [Path(__file__), Path("scripts/sac_bellman_trace.py"),
             Path("scripts/audit_sac_fixed_policy_critic.py"), Path("src/training/sac/networks.py"),
             Path("src/training/sac/policy.py"), Path("src/training/sac/replay.py"),
             Path("scripts/sac_diagnostic_actor.py"), Path("scripts/sac_diagnostic_critic.py")]
    result = dict(status="COMPLETE", qualified=False, scope="TRAIN_SAMPLED_PATH_VALUE_CALIBRATION",
                  actual_transitions=len(rows), complete_rollouts=len(episodes),
                  complete_ticks=len(all_complete), budget_truncated_excluded=excluded,
                  alpha=alpha, action_reproduction_tolerance=1e-9,
                  action_reproduction_max_difference=maximum_action_difference,
                  summaries=summaries, episodes=episodes, rows=rows,
                  input_sha256=checked["source_sha256"], source_sha256={str(p): sha(p) for p in paths},
                  note="Same actual stochastic actions and realized future returns for both critic snapshots. "
                       "Each Q target excludes entropy of its own conditional action. Budget-truncated tails excluded; "
                       "no invented terminal/bootstrap value. Correlated trajectory suffixes are not independent samples. "
                       "Training calibration, not generalization, exact expected Q, causal proof, or driving improvement.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "complete_rollouts", "complete_ticks", "summaries")}))


if __name__ == "__main__":
    main()
