#!/usr/bin/env python3
"""Add event rates and selected-path tracking RMS to a verified paired review."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sac_capture_metrics import episode_metrics, summarize


def review(root, output):
    if output.exists():
        raise FileExistsError(output)
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    parent = read(root / "full-validation-review.json")
    decision = read(root / "decision.json")
    if (parent["status"] != "COMPLETE" or not parent["initial_frames_maps_identical"]
            or parent["decision_sha256"] != sha(root / "decision.json")):
        raise ValueError("requires verified complete paired inputs")
    expected = {tuple(x) for x in decision["all_validation_pairs"]}
    cohorts = dict(decision["cohorts"])
    cohorts["rule_reference"] = [str(Path(decision["rule_reference"]) / "captures")]
    if set(cohorts) != set(parent["evaluations"]):
        raise ValueError("cohort coverage differs from verified parent")
    inputs = {str(root / n): sha(root / n) for n in ("decision.json", "full-validation-review.json")}
    results, initial = {}, {}
    for arm, directories in cohorts.items():
        episodes = []
        prior = {(e["scene_id"], e["provider"]): e for e in parent["evaluations"][arm]["episodes"]}
        for directory in map(Path, directories):
            report_path = directory / "report.json"
            if sha(report_path) != parent["inputs_sha256"][str(report_path)]:
                raise ValueError("evaluation report changed")
            inputs[str(report_path)] = sha(report_path)
            report = read(report_path)
            if report["status"] != "COMPLETE":
                raise ValueError("evaluation is incomplete")
            for record in report["episodes"]:
                path = directory / record["evidence"]
                if path.resolve().parent != directory.resolve():
                    raise ValueError("trace escapes evaluation directory")
                if sha(path) != parent["inputs_sha256"][str(path)]:
                    raise ValueError("actual trace changed")
                inputs[str(path)] = sha(path)
                trace = read(path)
                e = episode_metrics(trace)
                key = e["scene_id"], e["provider"]
                if key not in expected or key in {(v["scene_id"], v["provider"]) for v in episodes}:
                    raise ValueError("unexpected or duplicate scene/provider")
                if key in initial and initial[key] != (trace["frames"][0], trace["map"]):
                    raise ValueError("initial state/map differs across arms")
                initial[key] = trace["frames"][0], trace["map"]
                for field in ("outcome", "ticks", "rejected", "braking", "stationary"):
                    if e[field] != prior[key][field]:
                        raise ValueError(f"original metric differs: {field}")
                for field in ("initial_speed", "return_training"):
                    if not math.isclose(e[field], prior[key][field], rel_tol=0, abs_tol=1e-10):
                        raise ValueError(f"original metric differs: {field}")
                e["evidence"] = str(path)
                episodes.append(e)
        if {(e["scene_id"], e["provider"]) for e in episodes} != expected:
            raise ValueError("missing evaluation pairs")
        results[arm] = dict(all=summarize(episodes), episodes=episodes,
            by_provider={p: summarize([e for e in episodes if e["provider"] == p])
                         for p in sorted({e["provider"] for e in episodes})})
    baseline_path = Path(decision["rule_reference"]) / "baseline.json"
    if sha(baseline_path) != parent["inputs_sha256"][str(baseline_path)]:
        raise ValueError("original Rule runtime report changed")
    inputs[str(baseline_path)] = sha(baseline_path)
    baseline = read(baseline_path)
    if baseline["status"] != "COMPLETE" or set(baseline["by_behavior"]) != {"KEEP", "FOLLOW"}:
        raise ValueError("original Rule runtime evaluation is incomplete")
    for behavior, original in baseline["by_behavior"].items():
        derived = results["rule_reference"]["by_provider"]["fixed_" + behavior.lower()]
        for field in ("episodes", "ticks", "tracking_samples", "sources", "outcomes"):
            if derived[field] != original[field]:
                raise ValueError(f"Rule runtime denominator differs: {field}")
        if derived["event_ticks"] != original["events"]:
            raise ValueError("Rule runtime event flags differ")
        for field in ("success_rate", "collision_rate", "offroad_rate", "mean_return",
                      "mean_episode_ticks", "raw_rejection_rate", "braking_intervention_rate",
                      "stationary_fraction", "tracking_position_rms"):
            a, b = derived[field], original[field]
            if (a is None or b is None) and a is b:
                continue
            if a is None or b is None or not math.isclose(a, b, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Rule runtime metric differs: {field}")
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError("input changed during review")
    report = dict(status="COMPLETE", qualified=False,
        scope="SUPPLEMENTARY_EVENT_AND_SELECTED_TRAJECTORY_METRICS",
        created_kst=datetime.now().astimezone().isoformat(), evaluations=results,
        rule_runtime_metrics_verified=True,
        original_review_new_episodes_by_model=parent["new_episodes_by_model"],
        inputs_sha256=inputs, script_sha256=sha(Path(__file__)),
        metric_sha256=sha(Path(__file__).with_name("sac_capture_metrics.py")),
        new_policy_calls=0, new_learning_updates=0, replay_insertions=0,
        limits=["Repeated development traces and existing same-condition Rule reference.",
                "Tracking uses selected executed_xy[0] at the post-step time, not raw proposals.",
                "Ticks without a selected trajectory are excluded, not assigned zero error.",
                "No model selection, new rollout, independent-test or policy-stability claim."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.output)
    print(json.dumps({k: v["all"] for k, v in result["evaluations"].items()}, indent=2))
