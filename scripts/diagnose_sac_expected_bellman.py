#!/usr/bin/env python3
"""Read-only Monte Carlo Bellman diagnostics on fixed-policy training paths.

Common Gaussian samples compare online/target Q expectations at each actual
next observation. This is not exact integration or independent policy evaluation.
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
    for name in ("diagnostic", "source", "actual-values", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--draws", type=int, choices=(64, 256), default=64)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.audit_sac_fixed_policy_critic import audit
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
    report, source, actual = (read(p) for p in (args.diagnostic / "report.json",
                             args.source / "report.json", args.actual_values))
    assert actual["status"] == "COMPLETE" and actual["scope"] == "TRAIN_SAMPLED_PATH_VALUE_CALIBRATION"
    for p, expected in actual["input_sha256"].items():
        assert sha(Path(p)) == expected, p
    inputs = dict(checked["source_sha256"], **{str(args.actual_values): sha(args.actual_values)})
    script_paths = [Path(__file__), Path("scripts/sac_bellman_trace.py"),
                    Path("scripts/audit_sac_fixed_policy_critic.py"),
                    Path("scripts/sac_diagnostic_actor.py"), Path("scripts/sac_diagnostic_critic.py"),
                    Path("src/training/sac/networks.py"), Path("src/training/sac/replay.py")]
    sources = {str(p): sha(p) for p in script_paths}
    inherited = report["inherited_replay_size"]
    with np.load(args.diagnostic / "diagnostic-replay.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key][inherited:].copy() for key in
                  ("obs", "fields", "items", "next_obs", "next_fields", "next_items", "raw_u", "reward", "bootstrap")}
    complete = {ep["rollout"] for ep in actual["episodes"]}
    cohort = [row for row in actual["rows"] if row["rollout"] in complete]
    assert len(cohort) == actual["complete_ticks"]
    gpu = jax.devices("gpu")[0]
    results, maximum_reference_difference = [], 0.
    with jax.default_device(gpu):
        template = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                              actor=diagnostic_actor(source["arms"]["online"], restoring=True),
                              critic=diagnostic_critic(source["arms"]["online"], restoring=True))
        states = {}
        for name, path in (("before", args.source / "online-learner.msgpack"),
                           ("after", args.diagnostic / "critic-evaluation-state.msgpack")):
            states[name] = jax.device_put(serialization.from_bytes(template, path.read_bytes()), gpu)
            assert serialization.to_bytes(states[name]) == path.read_bytes()
        assert serialization.to_bytes(states["before"].actor) == serialization.to_bytes(states["after"].actor)
        state = states["before"]
        alpha = float(jnp.exp(state.log_alpha))
        assert alpha == actual["alpha"] == float(jnp.exp(states["after"].log_alpha))
        forward = jax.jit(state.actor.apply_fn)
        apply1, apply2 = state.q1.apply_fn, state.q2.apply_fn

        @jax.jit
        def values(p1, p2, obs, raw):
            return jnp.minimum(apply1({"params": p1}, obs, raw), apply2({"params": p2}, obs, raw))

        def obs(index, prefix="", size=1):
            return unpack_observation(*(jnp.repeat(jnp.asarray(arrays[prefix + k][index:index+1]), size, axis=0)
                                        for k in ("obs", "fields", "items")))

        for position, row in enumerate(cohort):
            i = row["collection_id"] - 1
            assert arrays["bootstrap"][i] == row["bootstrap"]
            assert arrays["reward"][i] == row["reward"]
            current = obs(i)
            raw = jnp.asarray(arrays["raw_u"][i:i+1])
            record = {k: row[k] for k in ("collection_id", "rollout", "provider", "branch_step", "bootstrap", "reward")}
            samples = {}
            for name, snapshot in states.items():
                # Preserve the scalar inference route used by actual-path analysis.
                for label, p1, p2 in (("q", snapshot.q1.params, snapshot.q2.params),
                                      ("target_q", snapshot.target1, snapshot.target2)):
                    estimates = np.asarray(values(p1, p2, current, raw))
                    difference = float(np.max(np.abs(estimates - row[name][label])))
                    assert difference <= 1e-3, (i, name, label, difference)
                    maximum_reference_difference = max(maximum_reference_difference, difference)
                samples[name] = {}
            if row["bootstrap"]:
                next_single = obs(i, "next_")
                mu, rho = forward({"params": state.actor.params}, next_single)
                mu, rho = (jnp.repeat(value, args.draws, axis=0) for value in (mu, rho))
                key = jax.random.fold_in(jax.random.PRNGKey(args.seed), i)
                action, ell, _ = sample_action(mu, rho, jnp.repeat(observation_speed(next_single), args.draws), key)
                following = next_single
                for name, snapshot in states.items():
                    for label, p1, p2 in (("online", snapshot.q1.params, snapshot.q2.params),
                                          ("target", snapshot.target1, snapshot.target2)):
                        estimates = np.asarray([float(np.asarray(values(p1, p2, following, action[j:j+1]))[0])
                                                for j in range(args.draws)]) - alpha * np.asarray(ell)
                        assert np.isfinite(estimates).all()
                        samples[name][label] = estimates
            for name in states:
                summary = dict(q=row[name]["q"], observed_soft_return=row[name]["observed_soft_return"])
                for label in ("online", "target"):
                    if row["bootstrap"]:
                        estimates = samples[name][label]
                        mean, std = float(estimates.mean()), float(estimates.std(ddof=1))
                        se = std / np.sqrt(args.draws)
                    else:
                        mean = std = se = 0.
                    summary[label] = dict(next_soft_value_mean=mean, next_soft_value_std=std,
                                          mc_standard_error=float(se),
                                          estimated_bellman_residual=row[name]["q"] - row["reward"] - .99 * mean,
                                          sampled_path_residual=row[name][label + "_bellman_residual"])
                summary["estimated_target_lag"] = .99 * (summary["target"]["next_soft_value_mean"] -
                                                           summary["online"]["next_soft_value_mean"])
                np.testing.assert_allclose(summary["online"]["estimated_bellman_residual"],
                    summary["target"]["estimated_bellman_residual"] + summary["estimated_target_lag"], rtol=0, atol=1e-10)
                record[name] = summary
            results.append(record)
            if (position + 1) % 100 == 0:
                print(json.dumps(dict(evaluated=position+1, total=len(cohort))), flush=True)
    for group in (inputs, sources):
        for p, expected in group.items():
            assert sha(Path(p)) == expected, p
    result = dict(status="COMPLETE", qualified=False, scope="MC_EXPECTED_BELLMAN_TRAIN_DIAGNOSTIC",
                  seed=args.seed, draws_per_nonterminal=args.draws, ticks=len(results),
                  next_observation_evaluations=sum(row["bootstrap"] for row in results),
                  scalar_reference_tolerance=1e-3, scalar_reference_max_difference=maximum_reference_difference,
                  critic_inference_batch_size=1,
                  alpha=alpha, rows=results, input_sha256=inputs, source_sha256=sources,
                  note="Same actual next observation; common Gaussian samples across snapshots and online/target Q. "
                       "Terminal next actor/Q never evaluated. MC standard errors describe sampling only, not critic accuracy. "
                       "Repeated training roots/correlated rows are not independent validation. No learning or policy promotion.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "ticks", "next_observation_evaluations", "scalar_reference_max_difference")}))


if __name__ == "__main__":
    main()
