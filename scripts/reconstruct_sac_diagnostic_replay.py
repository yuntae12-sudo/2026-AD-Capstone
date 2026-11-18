#!/usr/bin/env python3
"""Reproduce logged own-policy actions to recover audit observations, never train."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.collector import collect_step
    from src.training.sac.policy import proposal_from_action
    from src.training.sac.replay import Replay
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario

    source_path = args.diagnostic / "report.json"
    ticks_path = args.diagnostic / "ticks.jsonl"
    source = json.loads(source_path.read_text())
    version = versions(args.manifest)
    if source["status"] != "COMPLETE" or source["versions"] != version:
        raise ValueError("completed identical-runtime online diagnostic required")
    ticks = [json.loads(line) for line in ticks_path.read_text().splitlines()]
    assert [r["update"] for r in ticks] == list(range(1, source["updates"] + 1))
    assert len(ticks) == source["new_transitions"]
    paths = {
        (row["scene_id"], row["provider"]): path
        for row, path in selected_scenes(args.manifest, "train")
    }
    replay = Replay(len(ticks), versions=version)
    report = dict(
        status="RUNNING",
        qualified=False,
        scope="READ_ONLY_RECONSTRUCTED_ONLINE_EXPERIENCE",
        versions=version,
        diagnostic_sha256=file_hash(source_path),
        ticks_sha256=file_hash(ticks_path),
        script_sha256=file_hash(__file__),
        transitions=0,
        episodes=[],
        note="New online experience only, no warmup prefix, learner or replay insertion into any training run. Actual recorded raw actions are replayed; anchors, rewards, control and termination must match. Not a resumable training checkpoint.",
    )
    args.output.mkdir(parents=True)
    uid = None
    e = None
    last_finished = True
    try:
        for row in ticks:
            proposal = row["raw_proposal"]
            token = proposal["observation_token"]
            raw = np.asarray(proposal["raw_u"], np.float64)
            if token["episode_uid"] != uid:
                assert last_finished and row["tick"] == 1 and token["sim_tick"] == 0
                uid = token["episode_uid"]
                e = SACPlanningEnvironment()
                boundary = e.reset(
                    parse_scenario(
                        paths[(row["scene_id"], row["provider"])].read_bytes()
                    ),
                    seed=source["seed"],
                    provider_spec=row["provider"],
                )
                episode = dict(
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    episode_uid=uid,
                    ticks=0,
                    accepted=0,
                )
                report["episodes"].append(episode)
            assert e.tick == token["sim_tick"] and token["scene_id"] == row["scene_id"]
            for actual, expected in [
                (e.ego.position, proposal["anchor_world_xy"]),
                (e.ego.velocity, proposal["anchor_velocity_world"]),
                (e.ego.acceleration, proposal["anchor_acceleration_world"]),
                (e.ego.yaw, proposal["anchor_yaw"]),
            ]:
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
            step = collect_step(
                e, boundary, lambda env, tok: proposal_from_action(env, tok, raw)
            )
            if step.transition is None:
                raise ValueError("invalid reconstructed transition")
            result = step.result
            boundary = step.boundary
            assert (
                e.tick == row["tick"]
                and result.check_result.valid == row["check_valid"]
            )
            assert (
                result.check_result.reason == row["check_reason"]
                and result.source.value == row["source"]
            )
            assert (
                result.terminated == row["terminated"]
                and result.truncated == row["truncated"]
            )
            np.testing.assert_allclose(
                step.reward.training, row["reward"], atol=1e-8, rtol=0
            )
            for field in ("acceleration_mps2", "curvature_per_m"):
                np.testing.assert_allclose(
                    getattr(result.actual_control, field),
                    row["actual_control"][field],
                    atol=1e-8,
                    rtol=0,
                )
            np.testing.assert_array_equal(step.transition.raw_u, raw)
            replay.add(replace(step.transition, episode_uid=uid))
            report["transitions"] += 1
            episode["ticks"] += 1
            episode["accepted"] += int(result.check_result.valid)
            last_finished = result.terminated or result.truncated
            if last_finished or row["update"] % 100 == 0:
                write_report(args.output / "report.json", report)
                print(
                    json.dumps(
                        dict(
                            transitions=report["transitions"],
                            episodes=len(report["episodes"]),
                        )
                    ),
                    flush=True,
                )
        for restored, expected in zip(report["episodes"], source["episodes"]):
            for key in ("scene_id", "provider", "ticks", "accepted"):
                assert restored[key] == expected[key]
        assert len(report["episodes"]) == len(source["episodes"])
        assert (
            file_hash(source_path) == report["diagnostic_sha256"]
            and file_hash(ticks_path) == report["ticks_sha256"]
        )
        assert versions(args.manifest) == version
        replay.save(args.output / "replay.npz")
        report.update(
            status="COMPLETE", replay_sha256=file_hash(args.output / "replay.npz")
        )
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
