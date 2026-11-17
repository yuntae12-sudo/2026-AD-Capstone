#!/usr/bin/env python3
"""Freeze an online SAC trace and produce failure statistics plus verified replays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--replay-collisions", type=int, default=2)
    args = p.parse_args()
    if args.replay_collisions < 0:
        p.error("replay-collisions must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from scripts.sac_failure_analysis import (
        freeze_trace,
        analyze,
        select_collisions,
        episode_groups,
        evaluation_trend,
    )
    from scripts.sac_failure_report import write_markdown, plot_report

    metadata, snapshot = freeze_trace(args.run, args.output)
    if (
        metadata.get("training_mode") != "pure-sac"
        or metadata.get("phase") != "ONLINE_SAC"
    ):
        raise ValueError("this report requires the pure SAC online trace schema")
    if (
        hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        != metadata["versions"]["manifest_sha256"]
    ):
        raise ValueError("failure report manifest mismatch")
    report = analyze(args.output / "trace.snapshot.jsonl", metadata)
    report.update(
        status="ANALYZING",
        created_utc=datetime.now(timezone.utc).isoformat(),
        snapshot=snapshot,
        replays=[],
        replay_selection="earliest collision per behavior; then new behavior/speed stratum; then chronological",
        evaluation_trend=evaluation_trend(
            (args.run / "evaluations").glob("*.json"),
            metadata["versions"],
            metadata["seed"],
        ),
        analysis_source_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "analyze_sac_failures.py",
                "sac_failure_analysis.py",
                "sac_failure_replay.py",
                "sac_failure_report.py",
            )
        },
    )

    def save():
        temp = args.output / "analysis.json.tmp"
        temp.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        temp.replace(args.output / "analysis.json")

    selected = select_collisions(report["episodes"], args.replay_collisions)
    report["selected_episode_uids"] = [e["episode_uid"] for e in selected]
    save()
    if selected:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        import jax
        from src.training.sac.runtime import versions, selected_scenes
        from src.scenarios.womd20 import parse_scenario
        from scripts.sac_failure_replay import replay_episode

        if versions(args.manifest) != metadata["versions"]:
            raise ValueError("cannot replay a different runtime/manifest")
        candidates = {
            (r["scene_id"], r["provider"]): path
            for r, path in selected_scenes(args.manifest, "train")
        }
        ids = set(report["selected_episode_uids"])
        for rows in episode_groups(args.output / "trace.snapshot.jsonl"):
            first = rows[0]
            if first["episode_uid"] not in ids:
                continue
            try:
                with jax.default_device(jax.devices("cpu")[0]):
                    scenario = parse_scenario(
                        candidates[(first["scene_id"], first["provider"])].read_bytes()
                    )
                    case = replay_episode(rows, scenario, metadata["seed"])
            except Exception as exc:
                case = dict(
                    episode_uid=first["episode_uid"],
                    status="FAILED_VERIFICATION",
                    error=f"{type(exc).__name__}: {exc}",
                )
            report["replays"].append(case)
            save()
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in case.items()
                        if k not in ("contacts", "last_3s_kinematic_violations")
                    }
                ),
                flush=True,
            )
        if versions(args.manifest) != metadata["versions"]:
            raise RuntimeError("runtime changed during diagnostic replay")
    report["status"] = (
        "PARTIAL_REPLAY_FAILURE"
        if any(c["status"] != "VERIFIED" for c in report["replays"])
        else "COMPLETE"
    )
    save()
    plot_report(report, args.output)
    write_markdown(report, args.output / "report.md")
    print(
        json.dumps(
            dict(
                status=report["status"],
                summary=report["summary"],
                report=str(args.output / "report.md"),
            )
        ),
        flush=True,
    )
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
