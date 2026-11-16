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


def compare(reports):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    for name in ("manifest", "output"):
        evaluate_parser.add_argument("--" + name, type=Path, required=True)
    evaluate_parser.add_argument(
        "--policy", choices=("BC", "SAC", *BASELINES), required=True
    )
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    evaluate_parser.add_argument("--smoke", action="store_true")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--reports", type=Path, nargs="+", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare([json.loads(p.read_text()) for p in args.reports])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        lines = [
            "# BC / SAC validation comparison",
            "",
            "This is an initial experiment, not evidence of convergence or test-set performance.",
            "",
            "| Behavior | Metric | BC mean | SAC mean | SAC − BC |",
            "|---|---|---:|---:|---:|",
        ]
        for behavior, metrics in result["by_behavior"].items():
            for metric, values in metrics.items():
                lines.append(
                    f"| {behavior} | {metric} | {np.mean(values['bc_by_seed']):.6f} | {np.mean(values['sac_by_seed']):.6f} | {values['mean_delta']:.6f} |"
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
    if args.policy in ("BC", "SAC"):
        if args.checkpoint is None:
            parser.error("learned policy requires --checkpoint")
        with jax.default_device(device):
            state, meta = load_model(args.checkpoint, version, args.seed)
            state = jax.device_put(state, device)
        run = meta["run_state"]
        if args.policy == "BC":
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
    )
    if result["status"] != "COMPLETE":
        raise RuntimeError("evaluation did not complete")


if __name__ == "__main__":
    main()
