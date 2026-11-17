#!/usr/bin/env python3
"""Capture real paired Waymax rollouts and export an evidence-backed GIF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Visualization uses CPU only; the independent training process keeps its GPU.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    for name in ("manifest", "checkpoint", "output"):
        cap.add_argument("--" + name, type=Path, required=True)
    cap.add_argument("--scene", required=True)
    cap.add_argument(
        "--provider", choices=("fixed_keep", "fixed_follow"), default="fixed_keep"
    )
    cap.add_argument("--split", choices=("train", "validation"), default="validation")
    cap.add_argument("--max-ticks", type=int, default=200)
    cap.add_argument("--stride", type=int, default=1)
    render = sub.add_parser("render")
    render.add_argument("--evidence", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    if args.stride < 1:
        p.error("stride must be positive")
    if args.command == "render":
        from src.visualization.sac_gif import render

        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                render(json.loads(args.evidence.read_text()), args.output, args.stride)
            )
        )
        return
    if args.max_ticks < 1:
        p.error("max-ticks must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from src.scenarios.womd20 import parse_scenario
    from scripts.sac_experiment_runtime import (
        load_model,
        model_proposer,
        scripted_proposer,
    )
    from scripts.pure_sac_protocol import protocol_metadata
    from src.visualization.sac_rollout import capture, require_paired
    from src.visualization.sac_gif import render

    version = versions(args.manifest)
    selected = [
        (r, path)
        for r, path in selected_scenes(args.manifest, args.split)
        if r["scene_id"] == args.scene and r["provider"] == args.provider
    ]
    if len(selected) != 1:
        raise ValueError("scene/provider absent from selected split")
    scenario = parse_scenario(selected[0][1].read_bytes())
    checkpoint = args.checkpoint.resolve()
    meta = json.loads((checkpoint / "manifest.json").read_text())
    seed = meta["run_state"]["seed"]
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        state, meta = load_model(checkpoint, version, seed)
        state = jax.device_put(state, cpu)
        run = meta["run_state"]
        report = run.get("report", {})
        if (
            int(state.bc_updates) != 0
            or report.get("training_mode") != "pure-sac"
            or report.get("initial_replay_size") != 0
            or report.get("protocol") != protocol_metadata()
        ):
            raise ValueError(
                "visualization requires a provenance-verified pure SAC checkpoint"
            )
        updates = int(state.updates)
        label = (
            "Pure SAC / UNTRAINED (0 updates)"
            if updates == 0
            else f"Pure SAC / {updates:,} updates"
        )
        _, proposer = model_proposer(state.actor, seed, True, cpu)
        args.output.mkdir(parents=True)
        evidence = dict(
            scope="WAYMAX_VISUALIZATION_PREVIEW",
            qualified=False,
            status="CAPTURING",
            versions=version,
            seed=seed,
            split=args.split,
            checkpoint_sha256=file_hash(checkpoint / "manifest.json"),
            checkpoint=str(checkpoint),
            sac_updates=updates,
            bc_updates=int(state.bc_updates),
            device=str(cpu),
            policy_mode="deterministic_mean",
            visualization_sha256={
                str(path.relative_to(Path(__file__).resolve().parents[1])): file_hash(
                    path
                )
                for path in (
                    Path(__file__),
                    Path(__file__).resolve().parents[1]
                    / "src/visualization/sac_rollout.py",
                    Path(__file__).resolve().parents[1]
                    / "src/visualization/sac_gif.py",
                )
            },
            note="Actual control rollouts; CPU inference preview, not the GPU evaluation or training trace. Other actors replay WOMD logs.",
            episodes=[],
        )
        start = time.monotonic()
        write_report(args.output / "evidence.json", evidence)
        for name, policy in (
            ("Rule-based / Frenet", scripted_proposer("teacher")),
            (label, proposer),
        ):
            episode = capture(
                scenario, args.provider, policy, name, seed, args.max_ticks
            )
            evidence["episodes"].append(episode)
            write_report(args.output / "evidence.json", evidence)
            print(
                json.dumps(
                    dict(
                        policy=name,
                        outcome=episode["outcome"],
                        summary=episode["summary"],
                    )
                ),
                flush=True,
            )
    require_paired(evidence["episodes"])
    if versions(args.manifest) != version:
        raise RuntimeError("runtime changed during visualization")
    evidence.update(status="CAPTURED", elapsed_s=time.monotonic() - start)
    write_report(args.output / "evidence.json", evidence)
    summary = render(evidence, args.output / "comparison.gif", args.stride)
    write_report(args.output / "render.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
