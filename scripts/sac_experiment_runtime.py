"""Experiment orchestration helpers; model/environment version stays immutable.

Runner hashes are recorded separately from runtime. This lets an unchanged BC
model enter a new, verified experiment without relabeling its training data.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def runner_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in (
        "sac_experiment_runtime.py",
        "run_sac_online.py",
        "evaluate_sac_models.py",
        "pure_sac_protocol.py",
    ):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    digest.update((root.parent / "configs/experiments/pure_sac_v1.json").read_bytes())
    return digest.hexdigest()


def checkpoint_metadata(path, version):
    from src.training.sac.checkpoint import file_hash

    path = Path(path)
    meta = json.loads((path / "manifest.json").read_text())
    if meta["versions"] != version or meta["episode_boundary"] is not True:
        raise ValueError("checkpoint runtime/boundary mismatch")
    if set(meta["hashes"]) != {"learner.msgpack", "replay.npz"}:
        raise ValueError("incomplete checkpoint hashes")
    for name, digest in meta["hashes"].items():
        if file_hash(path / name) != digest:
            raise ValueError("checkpoint checksum mismatch")
    return meta


def load_model(path, version, seed):
    from flax import serialization
    from src.contracts.observation import CoreObservation
    from src.training.sac.learner import initialize
    from src.training.sac.policy import tensor_observation

    meta = checkpoint_metadata(path, version)
    if meta["run_state"]["seed"] != seed:
        raise ValueError("checkpoint seed mismatch")
    template = initialize(tensor_observation(CoreObservation.empty()), seed)
    state = serialization.from_bytes(
        template, (Path(path) / "learner.msgpack").read_bytes()
    )
    return state, meta


def validate_gate_coverage(report, expected):
    """Reject missing scenes, incomplete mean rollouts and selectively omitted probes."""
    if report.get("status") != "PASSED" or report.get("passed") is not True:
        raise ValueError("BC gate is not passed")
    episodes = report.get("episodes", [])
    actual = [(e["scene_id"], e["provider"]) for e in episodes]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("BC gate validation coverage mismatch")
    if tuple(report.get("probe_ticks", [])) != (0, 10, 30):
        raise ValueError("BC gate probe schedule mismatch")
    probes = []
    for e in episodes:
        if e["behavior"] != e["provider"].removeprefix("fixed_").upper():
            raise ValueError("BC gate behavior/provider mismatch")
        if not e["outcome"] or e["outcome"] in (
            "SMOKE_BATCH_BOUNDARY",
            "COLLECTION_ERROR",
            "DATA_CORRUPTION",
        ):
            raise ValueError("incomplete BC mean episode")
        if not 0 < e["ticks"] <= 200 or not 0 <= e["accepted"] <= e["ticks"]:
            raise ValueError("invalid BC mean counts")
        ticks = [t for t in (0, 10, 30) if t < e["ticks"]]
        if set(e["probe_observations"]) != {str(t) for t in ticks}:
            raise ValueError("BC gate omitted reachable probe state")
        probes.extend((e["scene_id"], e["behavior"], t) for t in ticks)
    actual_probes = [
        (r["scene_id"], r["behavior"], r["tick"]) for r in report.get("states", [])
    ]
    if len(actual_probes) != len(set(actual_probes)) or set(actual_probes) != set(
        probes
    ):
        raise ValueError("BC gate probe coverage mismatch")
    for behavior in ("KEEP", "FOLLOW"):
        selected = [e for e in episodes if e["behavior"] == behavior]
        ticks = sum(e["ticks"] for e in selected)
        accepted = sum(e["accepted"] for e in selected)
        if not ticks or accepted / ticks < 0.9:
            raise ValueError("BC deterministic rollout mean gate failed")


def require_bc_checkpoint(checkpoint, gate_path, synthetic_path, manifest, seed):
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.gates import require_training_gate
    from src.training.sac.runtime import versions, selected_scenes

    version = versions(manifest)
    meta = checkpoint_metadata(checkpoint, version)
    run = meta["run_state"]
    if (
        run["phase"] != "BC"
        or run.get("qualified") is not True
        or run["bc_updates"] < 5000
        or run["seed"] != seed
    ):
        raise ValueError("online SAC requires qualified BC5000 for the same seed")
    report = json.loads(Path(gate_path).read_text())
    require_training_gate(
        report, version, file_hash(Path(checkpoint) / "manifest.json")
    )
    validate_gate_coverage(
        report,
        [
            (r["scene_id"], r["provider"])
            for r, _ in selected_scenes(manifest, "validation")
        ],
    )
    synthetic = json.loads(Path(synthetic_path).read_text())
    if (
        report["synthetic_sha256"] != file_hash(synthetic_path)
        or synthetic["versions"] != version
        or synthetic.get("passed") is not True
        or synthetic.get("qualified") is not True
        or synthetic["cases"] != report["synthetic_cases"]
    ):
        raise ValueError("synthetic gate provenance mismatch")
    return version, meta


def model_proposer(actor, seed, deterministic, device):
    """GPU policy inference, CPU simulator; no mixed-device JIT arguments."""
    import jax
    from src.training.sac.policy import Policy, proposal_from_action

    with jax.default_device(device):
        if any(
            getattr(x, "device", None) != device
            for x in jax.tree_util.tree_leaves(actor.params)
        ):
            raise ValueError("policy parameters must be placed on the inference device")
        policy = Policy(actor, seed=seed, deterministic=deterministic)

    def propose(environment, token):
        with jax.default_device(device):
            raw = policy.action(environment._boundary().sac_observation)
        return proposal_from_action(environment, token, raw)

    return policy, propose


def scripted_proposer(name):
    from src.training.sac.teacher import FrenetTeacher
    from src.training.sac.policy import proposal_from_action

    teacher = FrenetTeacher()

    def propose(e, token):
        if name == "teacher":
            return teacher.propose(e, token).proposal
        if name == "stationary_proposal":
            xy = np.broadcast_to(e.ego.position, (30, 2)).copy()
        elif name == "braking_proposal":
            from src.environment.womd_control.control import advance

            virtual, acceleration = e.ego, e.memory.last_a
            points = []
            for _ in range(30):
                acceleration = float(
                    np.clip(
                        np.clip(-2.0, acceleration - 0.15, acceleration + 0.15),
                        max(-6.0, -virtual.speed / 0.1),
                        6.0,
                    )
                )
                virtual = advance(virtual, acceleration, e.memory.last_kappa)
                points.append(virtual.position)
            xy = np.array(points)
        elif name == "always_rejected":
            return proposal_from_action(e, token, np.full((30, 2), 1e4, np.float64))
        elif name == "collision_seeking":
            objects = list(e._objects().values())
            if not objects:
                return proposal_from_action(e, token, np.zeros((30, 2), np.float64))
            other = min(
                objects, key=lambda x: np.linalg.norm(x.position - e.ego.position)
            )
            # Current observations only; the ordinary checker is never bypassed.
            t = np.arange(1, 31) / 10
            xy = e.ego.position + (
                other.position + other.velocity * 3 - e.ego.position
            ) * (t[:, None] / 3)
        else:
            raise ValueError(name)
        return e.proposal_from_world(xy, token, model_version=f"baseline:{name}")

    return propose


def summarize_episodes(episodes):
    result = {}
    for behavior in ("KEEP", "FOLLOW"):
        rows = [r for r in episodes if r["behavior"] == behavior]
        ticks = sum(r["ticks"] for r in rows)
        if not rows or not ticks:
            continue
        events = sum((Counter(r["events"]) for r in rows), Counter())
        sources = sum((Counter(r["sources"]) for r in rows), Counter())
        tracking_count = sum(r["tracking_count"] for r in rows)
        result[behavior] = dict(
            episodes=len(rows),
            ticks=ticks,
            success_rate=sum(r["outcome"] == "ROUTE_GOAL" for r in rows) / len(rows),
            collision_rate=sum("COLLISION" in r["events"] for r in rows) / len(rows),
            offroad_rate=sum("OFFROAD" in r["events"] for r in rows) / len(rows),
            mean_return=float(np.mean([r["return_training"] for r in rows])),
            mean_episode_ticks=ticks / len(rows),
            raw_rejection_rate=1 - sum(r["accepted"] for r in rows) / ticks,
            braking_intervention_rate=(sources["BRAKE"] + sources["EMERGENCY"]) / ticks,
            stationary_fraction=sum(r["stationary_ticks"] for r in rows) / ticks,
            tracking_position_rms=(
                (sum(r["tracking_position_squared"] for r in rows) / tracking_count)
                ** 0.5
                if tracking_count
                else None
            ),
            tracking_samples=tracking_count,
            sources=dict(sources),
            events=dict(events),
            outcomes=dict(Counter(r["outcome"] for r in rows)),
        )
    return result


def evaluate(
    manifest,
    output,
    name,
    proposer,
    seed,
    cpu_device,
    policy_sha256=None,
    smoke=False,
    expected_version=None,
    checkpoint_sha256=None,
    training_evidence=None,
):
    """All validation scene/provider pairs; inference never populates training replay."""
    import jax
    from src.training.sac.runtime import selected_scenes, versions, write_report
    from src.training.sac.collector import collect_step
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    current = versions(manifest)
    if expected_version is not None and current != expected_version:
        raise ValueError("model/evaluation runtime changed")
    report = dict(
        status="RUNNING",
        scope="SMOKE" if smoke else "VALIDATION_PERFORMANCE",
        versions=current,
        runner_sha256=runner_hash(),
        policy=name,
        policy_sha256=policy_sha256,
        checkpoint_sha256=checkpoint_sha256,
        training_evidence=training_evidence,
        seed=seed,
        episodes=[],
        group_status="scenario_fallback_overlap_unverified",
    )
    candidates = selected_scenes(manifest, "validation")
    if smoke:
        candidates = [
            next(c for c in candidates if c[0]["provider"] == p)
            for p in ("fixed_keep", "fixed_follow")
        ]
    start = time.monotonic()
    try:
        for row, path in candidates:
            with jax.default_device(cpu_device):
                e = SACPlanningEnvironment()
                b = e.reset(
                    parse_scenario(path.read_bytes()),
                    seed=seed,
                    provider_spec=row["provider"],
                )
                if e._state.sim_trajectory.x.device != cpu_device:
                    raise AssertionError("simulator left its CPU device")
                episode = dict(
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    behavior=e.command.draft.maneuver.value,
                    ticks=0,
                    accepted=0,
                    return_training=0.0,
                    stationary_ticks=0,
                    initial_speed=e.ego.speed,
                    simulator_device=str(e._state.sim_trajectory.x.device),
                    events={},
                    sources={},
                    reasons={},
                    tracking_count=0,
                    tracking_position_squared=0.0,
                    outcome=None,
                )
                report["episodes"].append(episode)
                while True:
                    step = collect_step(e, b, proposer)
                    r, b = step.result, step.boundary
                    episode["ticks"] += 1
                    episode["accepted"] += int(r.check_result.valid)
                    episode["return_training"] += step.reward.training
                    episode["stationary_ticks"] += int(
                        r.sample_valid and e.ego.speed < 0.1
                    )
                    for key, values in (
                        ("events", r.event_flags),
                        ("sources", [r.source.value]),
                        ("reasons", [r.check_result.reason]),
                    ):
                        for value in values:
                            episode[key][value] = episode[key].get(value, 0) + 1
                    if r.tracking_error is not None:
                        episode["tracking_count"] += 1
                        episode["tracking_position_squared"] += (
                            r.tracking_error.position_m**2
                        )
                    if r.terminated or r.truncated:
                        episode["outcome"] = r.terminal_reason.value
                        break
                    if smoke and episode["ticks"] >= 2:
                        episode["outcome"] = "SMOKE_BATCH_BOUNDARY"
                        break
            report["by_behavior"] = summarize_episodes(report["episodes"])
            write_report(output, report)
            print(json.dumps(dict(evaluation=name, episode=episode)), flush=True)
        corrupt = any(r["outcome"] == "DATA_CORRUPTION" for r in report["episodes"])
        report.update(
            status="FAILED_DATA_INTEGRITY" if corrupt else "COMPLETE",
            qualified=not smoke and not corrupt,
        )
    except BaseException as exc:
        report.update(
            status="ERROR", qualified=False, error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - start
        if report["versions"] != versions(manifest):
            report.update(
                status="ERROR",
                qualified=False,
                error="runtime changed during evaluation",
            )
        write_report(output, report)
    return report


def restore_trace(path, committed):
    """Preserve interrupted trace tails before resuming a numerical checkpoint."""
    path = Path(path)
    if not path.exists():
        if committed:
            raise ValueError("checkpoint trace is missing")
        return
    if path.stat().st_size < committed:
        raise ValueError("checkpoint trace is incomplete")
    if path.stat().st_size > committed:
        import shutil

        with path.open("rb") as source, path.with_name(
            f"{path.stem}-uncommitted-{time.time_ns()}.jsonl"
        ).open("wb") as dest:
            source.seek(committed)
            shutil.copyfileobj(source, dest)
        with path.open("r+b") as stream:
            stream.truncate(committed)
