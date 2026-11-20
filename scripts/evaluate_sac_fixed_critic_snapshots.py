#!/usr/bin/env python3
"""Compare critic snapshots on the same completed frozen-policy training paths.

Fixed actor/alpha make the original realized returns reusable. This measures
training-cohort calibration, not independent evaluation or policy improvement.
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
    for name in ("diagnostic", "parent", "source", "actual-values", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.audit_sac_fixed_policy_continuation import audit
    from scripts.sac_bellman_trace import decompose
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from src.contracts.observation import CoreObservation
    from src.training.sac.learner import initialize
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import unpack_observation

    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    checked = audit(args.diagnostic, args.parent, args.source)
    report, parent, source = (read(p / "report.json") for p in (args.diagnostic, args.parent, args.source))
    actual = read(args.actual_values)
    assert actual["status"] == "COMPLETE" and actual["scope"] == "TRAIN_SAMPLED_PATH_VALUE_CALIBRATION"
    assert actual["qualified"] is False and actual["actual_transitions"] == 1000
    for p, expected in actual["input_sha256"].items():
        assert sha(Path(p)) == expected, p
    old_inputs = {Path(p).resolve(): h for p, h in actual["input_sha256"].items()}
    assert old_inputs[(args.parent / "report.json").resolve()] == sha(args.parent / "report.json")
    assert old_inputs[(args.parent / "diagnostic-replay.npz").resolve()] == sha(args.parent / "diagnostic-replay.npz")
    paths = [(0, args.source / "online-learner.msgpack"), (1000, args.parent / "critic-evaluation-state.msgpack")]
    paths += [(s["critic_steps"], args.diagnostic / s["state_file"]) for s in report["snapshots"]]
    inputs = dict(checked["source_sha256"], **{str(args.actual_values): sha(args.actual_values)})
    inputs.update({str(path): sha(path) for _, path in paths})
    inputs.update({str(args.parent / name): sha(args.parent / name) for name in ("report.json", "diagnostic-replay.npz")})
    source_paths = [Path(__file__), Path("scripts/sac_bellman_trace.py"),
                    Path("scripts/audit_sac_fixed_policy_continuation.py"),
                    Path("scripts/sac_diagnostic_actor.py"), Path("scripts/sac_diagnostic_critic.py"),
                    Path("src/training/sac/replay.py"), Path("src/training/sac/networks.py")]
    sources = {str(p): sha(p) for p in source_paths}
    with np.load(args.parent / "diagnostic-replay.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key][6000:].copy() for key in ("obs", "fields", "items", "raw_u", "reward", "bootstrap")}
    complete = {ep["rollout"] for ep in actual["episodes"]}
    cohort = [row for row in actual["rows"] if row["rollout"] in complete]
    assert len(cohort) == actual["complete_ticks"] == 947 and len(complete) == 21
    results = []
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        template = initialize(tensor_observation(CoreObservation.empty()), source["seed"],
                              actor=diagnostic_actor(source["arms"]["online"], restoring=True),
                              critic=diagnostic_critic(source["arms"]["online"], restoring=True))
        actor_bytes = temperature_bytes = None
        apply1, apply2 = template.q1.apply_fn, template.q2.apply_fn

        @jax.jit
        def values(p1, p2, observation, raw):
            return jnp.minimum(apply1({"params": p1}, observation, raw), apply2({"params": p2}, observation, raw))

        for count, path in paths:
            state = jax.device_put(serialization.from_bytes(template, path.read_bytes()), gpu)
            assert serialization.to_bytes(state) == path.read_bytes()
            current_actor = serialization.to_bytes(state.actor)
            current_temperature = serialization.to_bytes((state.log_alpha, state.alpha_optimizer))
            if count == 0:
                actor_bytes, temperature_bytes = current_actor, current_temperature
            assert current_actor == actor_bytes and current_temperature == temperature_bytes
            assert hashlib.sha256(actor_bytes).hexdigest() == parent["actor_sha256"]
            assert int(state.updates) == source["updates"] + count and int(state.bc_updates) == 0
            alpha = float(jnp.exp(state.log_alpha))
            assert alpha == actual["alpha"] == parent["alpha"]
            groups, difference = {}, 0.
            for row in cohort:
                i = row["collection_id"] - 1
                assert arrays["reward"][i] == row["reward"] and arrays["bootstrap"][i] == row["bootstrap"]
                observation = unpack_observation(*(jnp.asarray(arrays[k][i:i+1]) for k in ("obs", "fields", "items")))
                raw = jnp.asarray(arrays["raw_u"][i:i+1])
                q = float(np.asarray(values(state.q1.params, state.q2.params, observation, raw))[0])
                target = float(np.asarray(values(state.target1, state.target2, observation, raw))[0])
                if count in (0, 1000):
                    original = row["before" if count == 0 else "after"]
                    delta = max(abs(q-original["q"]), abs(target-original["target_q"]))
                    assert delta <= 1e-3, (count, i, delta)
                    difference = max(difference, delta)
                result = {k: row[k] for k in ("collection_id", "rollout", "provider", "branch_step", "reward", "ell", "bootstrap")}
                result.update(q=q, target_q=target)
                groups.setdefault(row["rollout"], []).append(result)
            episodes, rows = [], []
            for rollout, group in groups.items():
                entropy = [0.] + [r["ell"] for r in group[1:]]
                decomposition = decompose([r["reward"] for r in group], entropy,
                    [r["q"] for r in group], [r["target_q"] for r in group],
                    [r["bootstrap"] for r in group], alpha)
                episodes.append(dict(rollout=rollout, provider=group[0]["provider"], **decomposition["summary"]))
                for row, accounting in zip(group, decomposition["rows"]):
                    row.update(accounting)
                    original = actual["rows"][row["collection_id"]-1]["before"]["observed_soft_return"]
                    assert abs(row["observed_soft_return"] - original) <= 1e-10
                    row["steps_before_terminal"] = len(group)-1-row["branch_step"]
                    rows.append(row)

            def summarize(selected):
                answer = dict(ticks=len(selected))
                if selected:
                    error = np.asarray([r["q_minus_observed_return"] for r in selected])
                    answer.update(mean_error=float(error.mean()), mae=float(np.abs(error).mean()),
                                  max_absolute_error=float(np.abs(error).max()))
                return answer

            summaries = {}
            for provider in ("all", "fixed_keep", "fixed_follow"):
                selected = [r for r in rows if provider == "all" or r["provider"] == provider]
                summaries[provider] = dict(all_ticks=summarize(selected),
                    terminal=summarize([r for r in selected if not r["bootstrap"]]),
                    nonterminal=summarize([r for r in selected if r["bootstrap"]]),
                    first_stochastic_action=summarize([r for r in selected if r["branch_step"] == 0]),
                    distance_to_terminal={label: summarize([r for r in selected if lo <= r["steps_before_terminal"] <= hi])
                        for label, lo, hi in (("0", 0, 0), ("1-5", 1, 5), ("6-15", 6, 15), ("16-30", 16, 30), ("31+", 31, 1000))})
            results.append(dict(critic_steps=count, state_sha256=sha(path), reference_max_difference=difference if count in (0, 1000) else None,
                                summaries=summaries, episodes=episodes, rows=rows))
            print(json.dumps(dict(critic_steps=count, summaries=summaries["all"])), flush=True)
    for group in (inputs, sources):
        for p, expected in group.items():
            assert sha(Path(p)) == expected, p
    result = dict(status="COMPLETE", qualified=False, scope="FIXED_TRAIN_COHORT_CRITIC_CALIBRATION",
                  total_critic_steps=report["critic_steps"], cohort_complete_rollouts=21, cohort_ticks=947,
                  actor_and_temperature_unchanged=True, numerical_reference_tolerance=1e-3,
                  budget_truncated_excluded=actual["budget_truncated_excluded"], snapshots=results,
                  input_sha256=inputs, source_sha256=sources,
                  note="Same947 completed training transitions and realized frozen-policy soft returns at every snapshot. "
                       "Original budget-truncated53 excluded. Scalar critic inference preserves original comparison route. "
                       "Repeated/correlated training data; no exact expected Q, held-out validation, or policy improvement claim.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
