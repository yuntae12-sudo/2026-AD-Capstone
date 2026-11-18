#!/usr/bin/env python3
"""Read-only same-state candidate checks along reproduced diagnostic rollouts."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def explain_check(environment, trajectory):
    """Observe the original checker calls, returning their outputs unchanged.

    Patches are scoped to this synchronous diagnostic call; never use concurrently
    with another checker in the same process. No threshold or decision is replaced.
    """
    from unittest.mock import patch
    import numpy as np
    from src.environment.womd_control import control

    events = []
    phase = "reference"
    original_scene, original_pd = control._scene_checks, control.pd_control
    state, token, memory, old_plan = (
        environment._state,
        environment.token,
        environment.memory,
        environment.old_plan,
    )

    def scenes(poses, route, objects, elapsed):
        reasons = original_scene(poses, route, objects, elapsed)
        for i, reason in enumerate(reasons):
            if reason:
                pose = poses[i]
                events.append(
                    dict(
                        phase=phase,
                        reason=reason,
                        future_seconds=float(elapsed[i]),
                        position=pose.position.tolist(),
                        yaw=float(pose.yaw),
                        speed=pose.speed,
                    )
                )
                break
        return reasons

    def pd(*args, **kwargs):
        nonlocal phase
        phase = "virtual_pd"
        return original_pd(*args, **kwargs)

    with patch.object(control, "_scene_checks", scenes), patch.object(
        control, "pd_control", pd
    ):
        check = environment._check(trajectory, environment._check_objects())
    assert environment._state is state and environment.token == token
    assert environment.memory == memory and environment.old_plan is old_plan
    times = environment.tick * 0.1 + np.linspace(0, 3, 301)
    v, a = (trajectory.evaluate(times, order) for order in (1, 2))
    heading = np.array([np.cos(environment.ego.yaw), np.sin(environment.ego.yaw)])
    return dict(
        check=asdict(check),
        decision_phase="accepted" if check.valid else phase,
        scene_events=events,
        raw_abs_max=float(np.abs(trajectory.proposal.raw_u).max()),
        endpoint=trajectory.evaluate(times[-1]).tolist(),
        endpoint_forward_speed=float(v[-1] @ heading),
        grid_min_forward_speed=float(np.min(v @ heading)),
        grid_max_vector_acceleration=float(np.linalg.norm(a, axis=1).max()),
        grid_note="301 samples: diagnostic observations, not continuous bounds",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "evaluation", "manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--variant", default="online")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument(
        "--providers", nargs="+", default=["fixed_keep", "fixed_follow"]
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import numpy as np
    from flax import serialization
    from src.contracts.observation import CoreObservation
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.learner import initialize
    from src.training.sac.networks import Actor, Critic
    from src.training.sac.policy import tensor_observation, proposal_from_action
    from src.training.sac.collector import collect_step
    from src.training.sac.teacher import FrenetTeacher
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.environment.womd_control.control import Trajectory, advance
    from src.environment.womd_control.geometry import project, wrap
    from src.scenarios.womd20 import parse_scenario
    from scripts.sac_experiment_runtime import model_proposer

    version = versions(args.manifest)
    source = json.loads((args.diagnostic / "report.json").read_text())
    evaluation = json.loads((args.evaluation / "report.json").read_text())
    if any(
        d["status"] != "COMPLETE" or d["versions"] != version
        for d in (source, evaluation)
    ):
        raise ValueError("completed identical-runtime diagnostic/evaluation required")
    arm = source["arms"][args.variant]
    weights = args.diagnostic / f"{args.variant}-learner.msgpack"
    if not (
        arm["status"] == "COMPLETE"
        and file_hash(weights) == arm["learner_sha256"] == evaluation["learner_sha256"]
        and file_hash(args.diagnostic / "report.json")
        == evaluation["diagnostic_sha256"]
    ):
        raise ValueError("source/evaluation/weights mismatch")
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    with jax.default_device(gpu):
        initial = initialize(
            tensor_observation(CoreObservation.empty()),
            source["seed"],
            actor=Actor(
                mean_gain=arm["mean_gain"],
                mean_basis=arm["mean_basis"],
                head_layer_norm=arm["actor_head_layer_norm"],
            ),
            critic=Critic(zero_output=arm["zero_q"]),
        )
        trained = jax.device_put(
            serialization.from_bytes(initial, weights.read_bytes()), gpu
        )
        q1, q2 = jax.jit(trained.q1.apply_fn), jax.jit(trained.q2.apply_fn)
    if int(trained.updates) != arm["updates"] or int(trained.bc_updates) != 0:
        raise ValueError("unexpected learner counters")
    _, learned_propose = model_proposer(trained.actor, source["seed"], True, gpu)

    def zero_propose(e, token):
        return proposal_from_action(e, token, np.zeros((30, 2), np.float64))

    selected = [
        (row, path)
        for row, path in selected_scenes(args.manifest, "validation")
        if (not args.scenes or row["scene_id"] in args.scenes)
        and row["provider"] in args.providers
    ]
    if not selected:
        raise ValueError("no selected pairs")
    if args.scenes and set(args.scenes) != {row["scene_id"] for row, _ in selected}:
        raise ValueError("requested scene unavailable")
    report = dict(
        status="RUNNING",
        qualified=False,
        scope="SAME_STATE_REPLANNING_DIAGNOSTIC",
        versions=version,
        script_sha256=file_hash(__file__),
        diagnostic_sha256=file_hash(args.diagnostic / "report.json"),
        evaluation_sha256=file_hash(args.evaluation / "report.json"),
        learner_sha256=file_hash(weights),
        seed=source["seed"],
        episodes=[],
        note="No learner/replay changes. Candidates share current state, PD memory, objects and contract. Rule-based is a read-only counterfactual, never training data. Reproduced trajectories must match saved CPU/GPU evaluation. Sparse same-state probes and one-step tracking do not certify global stability.",
    )
    args.output.mkdir(parents=True)
    try:
        for row, path in selected:
            for label, proposer in (
                ("untrained", zero_propose),
                ("updated", learned_propose),
            ):
                expected_path = (
                    args.evaluation
                    / f"{label}-{row['scene_id']}-{row['provider']}.json"
                )
                expected = json.loads(expected_path.read_text())
                # Same probe ticks on both histories, chosen from the saved learned rollout.
                learned_frames = json.loads(
                    (
                        args.evaluation
                        / f"updated-{row['scene_id']}-{row['provider']}.json"
                    ).read_text()
                )["frames"]
                probe_ticks = {0, 10, 20, 30, 40, 50, 60}
                previous = "RESET"
                for frame in learned_frames[1:]:
                    if frame["reason"] != previous:
                        t = frame["proposal_tick"]
                        probe_ticks.update(max(0, t + offset) for offset in (-1, 0, 1))
                        previous = frame["reason"]
                with jax.default_device(cpu):
                    e = SACPlanningEnvironment()
                    boundary = e.reset(
                        parse_scenario(path.read_bytes()),
                        seed=source["seed"],
                        provider_spec=row["provider"],
                    )
                teacher = FrenetTeacher()
                episode = dict(
                    label=label,
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    expected_sha256=file_hash(expected_path),
                    ticks=[],
                    probes=[],
                    status="RUNNING",
                )
                report["episodes"].append(episode)
                for expected_frame in expected["frames"][1:]:
                    with jax.default_device(cpu):
                        before = e.ego
                        if e.tick in probe_ticks:
                            generated = {
                                "zero": zero_propose(e, e.token),
                                "learned": learned_propose(e, e.token),
                                "rule_based": teacher.propose(e, e.token).proposal,
                            }
                            probe = dict(
                                tick=e.tick,
                                speed=before.speed,
                                position=before.position.tolist(),
                                velocity=before.velocity.tolist(),
                                acceleration=before.acceleration.tolist(),
                                yaw=before.yaw,
                                memory=asdict(e.memory),
                                candidates={},
                            )
                            for name, proposal in generated.items():
                                probe["candidates"][name] = explain_check(
                                    e, Trajectory(proposal)
                                )
                            with jax.default_device(gpu):
                                obs = tensor_observation(boundary.sac_observation)
                                for name, proposal in generated.items():
                                    action = jax.device_put(proposal.raw_u[None], gpu)
                                    probe["candidates"][name]["learned_min_q"] = float(
                                        jax.numpy.minimum(
                                            q1(
                                                {"params": trained.q1.params},
                                                obs,
                                                action,
                                            ),
                                            q2(
                                                {"params": trained.q2.params},
                                                obs,
                                                action,
                                            ),
                                        )[0]
                                    )
                            episode["probes"].append(probe)
                        step = collect_step(e, boundary, proposer)
                        boundary, result = step.boundary, step.result
                        control = result.actual_control
                        predicted = advance(
                            before, control.acceleration_mps2, control.curvature_per_m
                        )
                        measured = e.ego
                        np.testing.assert_allclose(
                            measured.position,
                            expected_frame["ego"]["xy"],
                            rtol=0,
                            atol=1e-6,
                        )
                        np.testing.assert_allclose(
                            measured.velocity,
                            expected_frame["ego"]["velocity"],
                            rtol=0,
                            atol=1e-6,
                        )
                        if (
                            result.check_result.reason != expected_frame["reason"]
                            or result.source.value != expected_frame["source"]
                            or (
                                result.terminal_reason.value
                                if result.terminal_reason
                                else None
                            )
                            != expected_frame["outcome"]
                        ):
                            raise AssertionError(
                                "saved rollout decision/outcome mismatch"
                            )
                        _, route_position, tangent = project(
                            e.route.center, measured.position
                        )
                        offset = measured.position - route_position
                        lateral = tangent[0] * offset[1] - tangent[1] * offset[0]
                        episode["ticks"].append(
                            dict(
                                tick=e.tick,
                                check=asdict(result.check_result),
                                source=result.source.value,
                                actual_speed=measured.speed,
                                route_lateral_m=float(lateral),
                                tracking=(
                                    asdict(result.tracking_error)
                                    if result.tracking_error
                                    else None
                                ),
                                attempts=[asdict(a) for a in result.attempts],
                                model_position_error=float(
                                    np.linalg.norm(
                                        predicted.position - measured.position
                                    )
                                ),
                                model_speed_error=abs(predicted.speed - measured.speed),
                                model_heading_error=float(
                                    abs(wrap(predicted.yaw - measured.yaw))
                                ),
                                acceleration=control.acceleration_mps2,
                                curvature=control.curvature_per_m,
                            )
                        )
                    if result.terminated or result.truncated:
                        break
                if len(episode["ticks"]) != expected["summary"]["ticks"]:
                    raise AssertionError("saved episode length mismatch")
                episode.update(status="COMPLETE", outcome=result.terminal_reason.value)
                write_report(
                    args.output / f"{label}-{row['scene_id']}-{row['provider']}.json",
                    episode,
                )
                write_report(args.output / "report.json", report)
                print(
                    json.dumps(
                        dict(
                            label=label,
                            scene=row["scene_id"],
                            provider=row["provider"],
                            probes=len(episode["probes"]),
                            outcome=episode["outcome"],
                        )
                    ),
                    flush=True,
                )
        if versions(args.manifest) != version:
            raise RuntimeError("runtime changed during diagnostic")
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
