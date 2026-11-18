#!/usr/bin/env python3
"""Bounded diagnostic continuation, with actual-driving regression stops."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def regression_gate(baseline, candidate):
    """Rows keyed by scene/provider, never allow missing or duplicate episodes."""

    def keyed(rows):
        result = {(r["scene_id"], r["provider"]): r for r in rows}
        if not result or len(result) != len(rows):
            raise ValueError("empty or duplicate evaluation rows")
        return result

    old, new = keyed(baseline), keyed(candidate)
    if set(old) != set(new):
        raise ValueError("evaluation scene/provider mismatch")
    successes = [key for key, row in old.items() if row["outcome"] == "ROUTE_GOAL"]
    if not successes:
        raise ValueError("baseline must contain successful episodes")
    regressions = [key for key in successes if new[key]["outcome"] != "ROUTE_GOAL"]
    collisions = sum(r["outcome"] == "COLLISION" for r in new.values())
    previous_collisions = sum(r["outcome"] == "COLLISION" for r in old.values())
    return dict(
        passed=not regressions and collisions <= previous_collisions,
        lost_successes=regressions,
        collisions=collisions,
        baseline_collisions=previous_collisions,
        success=sum(r["outcome"] == "ROUTE_GOAL" for r in new.values()),
        episodes=len(new),
        note="Cost-control gate, not qualification or independent performance evidence.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "initial-diagnostic",
        "baseline-evaluation",
        "manifest",
        "checkpoint",
        "configuration",
        "reference-cases",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    initial = read(args.initial_diagnostic / "report.json")
    baseline = read(args.baseline_evaluation / "report.json")
    if (
        initial["status"] != "COMPLETE"
        or initial["updates"] != 100
        or baseline["status"] != "COMPLETE"
    ):
        raise ValueError("completed 100-update run and baseline required")
    if (
        initial["versions"] != baseline["versions"]
        or initial["critic_action_kernel_init"] != "zero"
    ):
        raise ValueError("runtime/initialization mismatch")
    if initial["configuration_sha256"] != digest(
        args.configuration / "report.json"
    ) or initial["source_checkpoint_sha256"] != digest(
        args.checkpoint / "manifest.json"
    ):
        raise ValueError("initial run input provenance mismatch")
    rows = [r for r in baseline["episodes"] if r["label"] == "untrained"]
    regression_gate(rows, rows)
    scripts = [
        Path(__file__),
        Path("scripts/diagnose_sac_online_updates.py"),
        Path("scripts/sac_critic_initialization.py"),
        Path("scripts/evaluate_sac_short_updates.py"),
        Path("scripts/diagnose_sac_q_warmup.py"),
        args.initial_diagnostic / "report.json",
        args.baseline_evaluation / "report.json",
        args.configuration / "report.json",
        args.checkpoint / "manifest.json",
        args.manifest,
        args.reference_cases,
    ]
    frozen = {str(path): digest(path) for path in scripts}
    report = dict(
        status="RUNNING",
        qualified=False,
        scope="BOUNDED_INITIALIZATION_TRIAL",
        versions=initial["versions"],
        initial_sha256=digest(args.initial_diagnostic / "report.json"),
        baseline_sha256=digest(args.baseline_evaluation / "report.json"),
        source_sha256=frozen,
        stage="STARTING",
        executions=[],
        gates=[],
        note="Fixed conditional jobs only. Does not edit code, resume production, choose new hypotheses, or authorize adoption. A failed driving gate stops expansion; successful completion still needs review.",
    )
    args.output.mkdir(parents=True)
    report_path = args.output / "report.json"

    def run(stage, script, arguments, result_path):
        for path, checksum in frozen.items():
            if digest(path) != checksum:
                raise ValueError("source changed during bounded trial")
        command = [sys.executable, script, *map(str, arguments)]
        report["stage"] = stage
        entry = dict(stage=stage, command=command, status="RUNNING")
        report["executions"].append(entry)
        write(report_path, report)
        with (args.output / f"{stage}.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=7200
            )
        entry["returncode"] = result.returncode
        if result.returncode or read(result_path)["status"] != "COMPLETE":
            raise RuntimeError(f"{stage} failed; inspect its log")
        entry.update(status="COMPLETE", report_sha256=digest(result_path))
        write(report_path, report)

    try:
        current = args.initial_diagnostic
        for budget in (100, 1000):
            if budget == 1000:
                current = args.output / "online1000"
                run(
                    "train1000",
                    "scripts/diagnose_sac_online_updates.py",
                    [
                        "--checkpoint",
                        args.checkpoint,
                        "--configuration",
                        args.configuration,
                        "--variant",
                        args.variant,
                        "--manifest",
                        args.manifest,
                        "--output",
                        current,
                        "--updates",
                        "1000",
                        "--mean-gain",
                        initial["arms"]["online"]["mean_gain"],
                        "--zero-critic-action-init",
                        "--save-replay",
                    ],
                    current / "report.json",
                )
            evaluation = args.output / f"evaluation{budget}"
            run(
                f"evaluate{budget}",
                "scripts/evaluate_sac_short_updates.py",
                [
                    "--diagnostic",
                    current,
                    "--variant",
                    "online",
                    "--labels",
                    "updated",
                    "--manifest",
                    args.manifest,
                    "--output",
                    evaluation,
                    "--scenes",
                    *sorted({r["scene_id"] for r in rows}),
                ],
                evaluation / "report.json",
            )
            result = read(evaluation / "report.json")
            assert result["versions"] == baseline["versions"]
            assert result["diagnostic_sha256"] == digest(current / "report.json")
            assert (
                result["learner_sha256"]
                == read(current / "report.json")["arms"]["online"]["learner_sha256"]
            )
            candidate = result["episodes"]
            gate = regression_gate(rows, candidate)
            for row in candidate:
                old = next(
                    r
                    for r in rows
                    if (r["scene_id"], r["provider"])
                    == (row["scene_id"], row["provider"])
                )
                a, b = read(args.baseline_evaluation / old["evidence"]), read(
                    evaluation / row["evidence"]
                )
                assert (a["scene_id"], a["provider"], a["seed"]) == (
                    b["scene_id"],
                    b["provider"],
                    b["seed"],
                )
                assert a["frames"][0] == b["frames"][0] and a["map"] == b["map"]
            report["gates"].append(dict(updates=budget, **gate))
            write(report_path, report)
            if not gate["passed"]:
                report.update(
                    status="COMPLETE_WITH_REGRESSION", stage=f"STOPPED_AFTER_{budget}"
                )
                return
        run(
            "fixed_q_cases",
            "scripts/diagnose_sac_q_warmup.py",
            [
                "--checkpoint",
                args.checkpoint,
                "--manifest",
                args.manifest,
                "--diagnostics",
                current,
                "--reference-cases",
                args.reference_cases,
                "--output",
                args.output / "fixed-q-cases.json",
            ],
            args.output / "fixed-q-cases.json",
        )
        evaluation = args.output / "full-validation"
        run(
            "full_validation",
            "scripts/evaluate_sac_short_updates.py",
            [
                "--diagnostic",
                current,
                "--variant",
                "online",
                "--labels",
                "updated",
                "--manifest",
                args.manifest,
                "--output",
                evaluation,
            ],
            evaluation / "report.json",
        )
        report.update(status="COMPLETE_PENDING_REVIEW", stage="FINISHED")
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(report_path, report)


if __name__ == "__main__":
    main()
