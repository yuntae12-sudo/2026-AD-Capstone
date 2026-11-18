#!/usr/bin/env python3
"""Summarize completed frozen-Q probes without treating Q as a safety score."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(report):
    if report["status"] != "COMPLETE" or any(
        e["status"] != "COMPLETE" for e in report["episodes"]
    ):
        raise ValueError("complete audit required")
    rows, groups, continuations, rankings = [], [], [], []
    i = np.arange(1, 31)[:, None]
    j = np.arange(1, 31)[None, :]
    basis = np.where(
        j <= i, 0.1**3 * (i - j + 1) * (i - j + 2) / (2 * 3 * (i * 0.1) ** 2), 0
    )
    for episode in report["episodes"]:
        for probe in episode["probes"]:
            identity = dict(
                scene_id=episode["scene_id"],
                provider=episode["provider"],
                tick=probe["tick"],
            )
            rejected_groups = defaultdict(list)
            for name, c in probe["candidates"].items():
                kind = "sample" if name.startswith("sample_") else name
                initial_epsilon = np.linalg.solve(basis, np.asarray(c["raw_u"])) / (
                    0.02 * np.clip((probe["speed"] / 2) ** 3, 0.002, 1)
                )
                rows.append(
                    dict(
                        **identity,
                        name=name,
                        kind=kind,
                        speed=probe["speed"],
                        q1=c["q1"],
                        q2=c["q2"],
                        min_q=c["min_q"],
                        accepted=c["check"]["valid"],
                        reason=c["check"]["reason"],
                        source=c["source"],
                        reward=c["reward"]["training"],
                        outcome=c["outcome"],
                        bootstrap=c["bootstrap"],
                        q_minus_target=c["q_minus_target"],
                        policy_sigma_distance_rms=c["policy_sigma_distance_rms"],
                        initial_noise_distance_rms=float(
                            np.sqrt(np.mean(initial_epsilon**2))
                        ),
                    )
                )
                if not c["check"]["valid"]:
                    key = (
                        c["next_observation_sha256"],
                        c["reward"]["training"],
                        c["bootstrap"],
                        c["delta_ticks"],
                        json.dumps(c["actual_control"], sort_keys=True),
                    )
                    rejected_groups[key].append(name)
                for trial in probe["continuations"].get(name, []):
                    continuations.append(
                        dict(
                            **identity,
                            name=name,
                            q=c["min_q"],
                            q_minus_soft_return=c["min_q"] - trial["soft_return"],
                            **trial,
                        )
                    )
            for names in rejected_groups.values():
                if len(names) < 2:
                    continue
                values = [probe["candidates"][n]["min_q"] for n in names]
                policy_names = [
                    n for n in names if n == "mean" or n.startswith("sample_")
                ]
                policy_values = [probe["candidates"][n]["min_q"] for n in policy_names]
                groups.append(
                    dict(
                        **identity,
                        names=names,
                        size=len(names),
                        q_min=min(values),
                        q_max=max(values),
                        q_span=max(values) - min(values),
                        on_policy_names=policy_names,
                        on_policy_q_span=(
                            max(policy_values) - min(policy_values)
                            if len(policy_values) > 1
                            else None
                        ),
                        bootstrap=probe["candidates"][names[0]]["bootstrap"],
                    )
                )
            current = {
                n: c
                for n, c in probe["candidates"].items()
                if n == "mean" or n.startswith("sample_")
            }
            valid = [n for n, c in current.items() if c["check"]["valid"]]
            rejected = [n for n, c in current.items() if not c["check"]["valid"]]
            top = max(current, key=lambda n: current[n]["min_q"])
            rankings.append(
                dict(
                    **identity,
                    top=top,
                    top_valid=current[top]["check"]["valid"],
                    valid_count=len(valid),
                    rejected_count=len(rejected),
                    top_invalid_with_valid_alternative=bool(valid)
                    and not current[top]["check"]["valid"],
                    mean_invalid_with_valid_alternative=bool(valid)
                    and not current["mean"]["check"]["valid"],
                )
            )
    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        selected = [r for r in rows if r["kind"] == kind]
        terminal = [r for r in selected if not r["bootstrap"]]
        by_kind[kind] = dict(
            count=len(selected),
            accepted=sum(r["accepted"] for r in selected),
            reasons=dict(Counter(r["reason"] for r in selected)),
            q_mean=float(np.mean([r["min_q"] for r in selected])),
            mean_abs_one_step_residual=float(
                np.mean([abs(r["q_minus_target"]) for r in selected])
            ),
            terminal_count=len(terminal),
            terminal_q_mean=(
                float(np.mean([r["min_q"] for r in terminal])) if terminal else None
            ),
            terminal_mean_abs_error=(
                float(np.mean([abs(r["min_q"] - r["reward"]) for r in terminal]))
                if terminal
                else None
            ),
        )
    stats = dict(
        scope="DIAGNOSTIC_Q_AUDIT_NOT_PERFORMANCE_CERTIFICATION",
        qualified=False,
        episodes=len(report["episodes"]),
        unique_scenes=len({e["scene_id"] for e in report["episodes"]}),
        reproduced_ticks=sum(e["reproduced_ticks"] for e in report["episodes"]),
        states=len(rankings),
        candidates=len(rows),
        by_kind=by_kind,
        states_with_valid_on_policy_candidate=sum(
            r["valid_count"] > 0 for r in rankings
        ),
        mixed_on_policy_states=sum(
            r["valid_count"] > 0 and r["rejected_count"] > 0 for r in rankings
        ),
        top_invalid_with_valid_alternative=sum(
            r["top_invalid_with_valid_alternative"] for r in rankings
        ),
        mean_invalid_with_valid_alternative=sum(
            r["mean_invalid_with_valid_alternative"] for r in rankings
        ),
        equivalent_rejected_groups=len(groups),
        largest_equivalent_rejected_q_span=max(
            (g["q_span"] for g in groups), default=0
        ),
        equivalent_rejected_groups_with_multiple_policy_candidates=sum(
            len(g["on_policy_names"]) > 1 for g in groups
        ),
        largest_equivalent_rejected_on_policy_q_span=max(
            (
                g["on_policy_q_span"]
                for g in groups
                if g["on_policy_q_span"] is not None
            ),
            default=0,
        ),
        continuation_runs=len(continuations),
        continuation_outcomes=dict(Counter(c["outcome"] for c in continuations)),
        continuation_mean_q_error=(
            float(np.mean([c["q_minus_soft_return"] for c in continuations]))
            if continuations
            else None
        ),
        note="Terminal Q errors compare to known reward with zero bootstrap. Other one-step residuals use learned target Q and are not calibration ground truth. Continuation returns, when requested, use the configured small number of RNG replicates; no confidence interval or population claim. Initial-noise distance is a coordinate diagnostic, not a replay support test. States, candidates, command variants and RNG replicates are not independent scenes.",
    )
    return stats, dict(
        candidates=rows,
        equivalent_rejections=groups,
        continuations=continuations,
        rankings=rankings,
    )


def plots(tables, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terminal = [
        r for r in tables["candidates"] if r["name"] == "mean" and not r["bootstrap"]
    ]
    if terminal:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(terminal))
        ax.bar(
            x - 0.18,
            [r["min_q"] for r in terminal],
            0.36,
            label="Predicted min(Q1, Q2)",
        )
        ax.bar(
            x + 0.18,
            [r["reward"] for r in terminal],
            0.36,
            label="Actual terminal reward",
        )
        ax.set_xticks(
            x,
            [
                f"{r['scene_id'][:4]} / {r['provider'].removeprefix('fixed_')}\ntick {r['tick']}"
                for r in terminal
            ],
        )
        ax.set_ylabel("Reward units (terminal bootstrap = 0)")
        ax.set_title("Known terminal targets on selected validation states")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"terminal-q.{suffix}", dpi=160)
        plt.close(fig)
    grouped = defaultdict(list)
    for r in tables["continuations"]:
        grouped[(r["scene_id"], r["provider"], r["tick"], r["name"])].append(r)
    if grouped:
        fig, ax = plt.subplots(figsize=(10, max(5, len(grouped) * 0.4)))
        labels = []
        for y, (key, rows) in enumerate(grouped.items()):
            values = [r["soft_return"] for r in rows]
            ax.plot([min(values), max(values)], [y, y], color="#888888", lw=2)
            ax.scatter(
                values,
                [y] * len(values),
                color="#cf6a37",
                s=28,
                label="Actual stochastic suffix returns" if y == 0 else None,
            )
            ax.scatter(
                [rows[0]["q"]],
                [y],
                color="#2465ba",
                marker="x",
                s=55,
                label="Predicted min Q" if y == 0 else None,
            )
            labels.append(f"{key[0][:4]} / t{key[2]} / {key[3]}")
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Discounted soft return / Q")
        ax.set_title(
            "Frozen policy continuations: observed RNG replicates\nLines show observed ranges, not confidence intervals"
        )
        ax.legend(loc="lower left")
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"continuation-q.{suffix}", dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    stats, tables = summarize(report)
    stats.update(
        report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=report["versions"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.output / "tables.json").write_text(json.dumps(tables, indent=2) + "\n")
    for name, rows in tables.items():
        if rows:
            with (args.output / (name + ".csv")).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    plots(tables, args.output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
