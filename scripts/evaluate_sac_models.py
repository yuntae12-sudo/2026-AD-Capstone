#!/usr/bin/env python3
"""Evaluate BC/SAC/scripted policies on identical held-out validation scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASELINES = (
    "teacher",
    "stationary_proposal",
    "braking_proposal",
    "always_rejected",
    "collision_seeking",
)
METRICS = (
    "success_rate",
    "collision_rate",
    "offroad_rate",
    "mean_return",
    "mean_episode_ticks",
    "raw_rejection_rate",
    "braking_intervention_rate",
    "stationary_fraction",
)


def compare(reports, experiment="bc-sac"):
    """Require paired seeds/scenes; report effects without claiming significance."""
    entries = {}
    expected_scenes = None
    expected_versions = None
    for report in reports:
        if (
            report.get("scope") != "VALIDATION_PERFORMANCE"
            or report.get("status") != "COMPLETE"
            or report.get("qualified") is not True
        ):
            raise ValueError("comparison requires complete non-smoke evaluation")
        key = report["policy"], report["seed"]
        if key in entries:
            raise ValueError("duplicate policy/seed report")
        scenes = {(r["scene_id"], r["provider"]) for r in report["episodes"]}
        if len(scenes) != len(report["episodes"]):
            raise ValueError("duplicated evaluation episode")
        if expected_scenes is not None and (
            scenes != expected_scenes or report["versions"] != expected_versions
        ):
            raise ValueError("unpaired scenes/runtime in comparison")
        expected_scenes, expected_versions = scenes, report["versions"]
        entries[key] = report
    if experiment == "pure-sac":
        return compare_pure(entries, expected_versions)
    if experiment != "bc-sac":
        raise ValueError("unknown comparison experiment")
    if not all(
        (policy, seed) in entries for policy in ("BC", "SAC") for seed in (0, 1, 2)
    ):
        raise ValueError("comparison needs BC and SAC for seeds0/1/2")
    if not all((policy, 0) in entries for policy in BASELINES):
        raise ValueError("comparison needs all five scripted baselines")
    result = dict(
        scope="PAIRED_VALIDATION_COMPARISON",
        versions=expected_versions,
        by_behavior={},
        limitation="Development validation only; 3 seeds, scenario-fallback groups with unverified drive overlap. No test/convergence/safety claim.",
    )
    for behavior in ("KEEP", "FOLLOW"):
        result["by_behavior"][behavior] = {}
        for metric in METRICS:
            before = np.array(
                [entries[("BC", s)]["by_behavior"][behavior][metric] for s in (0, 1, 2)]
            )
            after = np.array(
                [
                    entries[("SAC", s)]["by_behavior"][behavior][metric]
                    for s in (0, 1, 2)
                ]
            )
            delta = after - before
            result["by_behavior"][behavior][metric] = dict(
                bc_by_seed=before.tolist(),
                sac_by_seed=after.tolist(),
                sac_minus_bc_by_seed=delta.tolist(),
                mean_delta=float(delta.mean()),
                sample_std_delta=float(delta.std(ddof=1)),
            )
    result["baselines"] = {
        f"{p}:seed{s}": r["by_behavior"]
        for (p, s), r in entries.items()
        if p in BASELINES
    }
    return result


def compare_pure(entries, versions):
    from scripts.pure_sac_protocol import protocol_metadata

    if not all(("PURE_SAC", s) in entries for s in (0, 1, 2)):
        raise ValueError("pure SAC comparison needs seeds0/1/2")
    if ("teacher", 0) not in entries:
        raise ValueError("pure SAC comparison needs the rule-based Frenet baseline")
    if any(p not in (*BASELINES, "PURE_SAC") for p, _ in entries):
        raise ValueError("BC or BC-SAC cannot enter the pure SAC main comparison")
    for s in (0, 1, 2):
        evidence = entries[("PURE_SAC", s)].get("training_evidence", {})
        if (
            evidence.get("protocol") != protocol_metadata()
            or evidence.get("bc_updates") != 0
            or evidence.get("initial_replay_size") != 0
            or evidence.get("updates", 0) < 10000
            or not evidence.get("checkpoint_sha256")
            or evidence.get("checkpoint_sha256")
            != entries[("PURE_SAC", s)].get("checkpoint_sha256")
        ):
            raise ValueError("missing pure SAC training provenance")
    result = dict(
        scope="RULE_BASED_VS_PURE_SAC_VALIDATION",
        versions=versions,
        protocol=protocol_metadata(),
        rule_based_implementation="FrenetTeacher (same checker and PD)",
        by_behavior={},
        limitation="Development validation only; 3 SAC seeds against one deterministic Frenet baseline. Scenario-fallback groups with unverified drive overlap. No test/convergence/safety claim.",
    )
    for behavior in ("KEEP", "FOLLOW"):
        result["by_behavior"][behavior] = {}
        for metric in METRICS:
            baseline = entries[("teacher", 0)]["by_behavior"][behavior][metric]
            learned = np.array(
                [
                    entries[("PURE_SAC", s)]["by_behavior"][behavior][metric]
                    for s in (0, 1, 2)
                ]
            )
            delta = learned - baseline
            result["by_behavior"][behavior][metric] = dict(
                rule_based=baseline,
                pure_sac_by_seed=learned.tolist(),
                pure_sac_minus_rule_based_by_seed=delta.tolist(),
                mean_delta=float(delta.mean()),
                sample_std_delta=float(delta.std(ddof=1)),
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    for name in ("manifest", "output"):
        evaluate_parser.add_argument("--" + name, type=Path, required=True)
    evaluate_parser.add_argument(
        "--policy", choices=("BC", "SAC", "PURE_SAC", *BASELINES), required=True
    )
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    evaluate_parser.add_argument("--smoke", action="store_true")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--reports", type=Path, nargs="+", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument(
        "--experiment", choices=("pure-sac", "bc-sac"), default="bc-sac"
    )
    args = parser.parse_args()
    if args.command == "compare":
        result = compare(
            [json.loads(p.read_text()) for p in args.reports], args.experiment
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        pure = args.experiment == "pure-sac"
        labels = ("Rule-based", "Pure SAC") if pure else ("BC", "SAC")
        lines = [
            f"# {labels[0]} / {labels[1]} validation comparison",
            "",
            "This is an initial experiment, not evidence of convergence or test-set performance.",
            "",
            f"| Behavior | Metric | {labels[0]} | {labels[1]} mean | Difference |",
            "|---|---|---:|---:|---:|",
        ]
        for behavior, metrics in result["by_behavior"].items():
            for metric, values in metrics.items():
                before = values["rule_based"] if pure else np.mean(values["bc_by_seed"])
                after = np.mean(
                    values["pure_sac_by_seed"] if pure else values["sac_by_seed"]
                )
                lines.append(
                    f"| {behavior} | {metric} | {before:.6f} | {after:.6f} | {values['mean_delta']:.6f} |"
                )
        lines.extend(
            [
                "",
                result["limitation"],
                "",
                "Review collision, rejection, braking, stationary fraction and return together; completion does not imply improvement.",
            ]
        )
        args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
        return
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    from flax import serialization
    from src.training.sac.runtime import versions
    from src.training.sac.checkpoint import file_hash
    from scripts.sac_experiment_runtime import (
        evaluate,
        load_model,
        model_proposer,
        scripted_proposer,
    )

    cpu = jax.devices("cpu")[0]
    device = jax.devices()[0]
    version = versions(args.manifest)
    sha = None
    training_evidence = None
    if args.policy in ("BC", "SAC", "PURE_SAC"):
        if args.checkpoint is None:
            parser.error("learned policy requires --checkpoint")
        with jax.default_device(device):
            state, meta = load_model(args.checkpoint, version, args.seed)
            state = jax.device_put(state, device)
        run = meta["run_state"]
        if args.policy == "PURE_SAC":
            from scripts.pure_sac_protocol import validate_pure_evaluation_checkpoint

            validate_pure_evaluation_checkpoint(state, run)
            training_evidence = dict(
                protocol=run["report"]["protocol"],
                bc_updates=int(state.bc_updates),
                initial_replay_size=run["report"]["initial_replay_size"],
                updates=int(state.updates),
                checkpoint_sha256=file_hash(args.checkpoint / "manifest.json"),
            )
        elif args.policy == "BC":
            if (
                run["phase"] != "BC"
                or int(state.updates) != 0
                or (
                    not args.smoke
                    and (not run["qualified"] or int(state.bc_updates) < 5000)
                )
            ):
                raise ValueError("not a qualified BC-only checkpoint")
        elif run["phase"] != "ONLINE_SAC" or int(state.updates) < 10000:
            raise ValueError("not a completed initial SAC experiment")
        sha = hashlib.sha256(serialization.to_bytes(state.actor.params)).hexdigest()
        _, propose = model_proposer(state.actor, args.seed, True, device)
    else:
        propose = scripted_proposer(args.policy)
    result = evaluate(
        args.manifest,
        args.output,
        args.policy,
        propose,
        args.seed,
        cpu,
        sha,
        args.smoke,
        expected_version=version,
        checkpoint_sha256=(
            file_hash(args.checkpoint / "manifest.json") if args.checkpoint else None
        ),
        training_evidence=training_evidence,
    )
    if result["status"] != "COMPLETE":
        raise RuntimeError("evaluation did not complete")


if __name__ == "__main__":
    main()
