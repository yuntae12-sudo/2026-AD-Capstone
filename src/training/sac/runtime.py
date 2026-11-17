"""Versioned data selection and long-running collection/BC entry points."""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import platform
import resource
import shutil
from pathlib import Path
import time
import numpy as np
from .checkpoint import file_hash, save_checkpoint, load_checkpoint
from .collector import collect_step
from .learner import initialize, bc_update
from .policy import tensor_observation
from .replay import Replay
from .teacher import FrenetTeacher
from src.contracts.observation import CoreObservation
from src.contracts.planning import to_wire
from src.environment.womd_control.environment import SACPlanningEnvironment
from src.scenarios.womd20 import parse_scenario

ROOT = Path(__file__).resolve().parents[3]


def write_report(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def versions(manifest):
    paths = []
    for folder in (
        "src/contracts",
        "src/environment/womd_control",
        "src/training/sac",
        "src/planning",
    ):
        paths.extend((ROOT / folder).glob("*.py"))
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes())
    return dict(
        schema=3,
        basis="1.2",
        learner_stability="critic_head_layer_norm_float64_gradient_norm_v1",
        follow_gap_semantics="reward_target_with_hard_physical_obb_order_v1",
        basis_sha256="10b2a882e5dfb3f31b4b300793a67a3d10aa19886421a3782e7b9a58394b4a64",
        python=platform.python_version(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "optax", "numpy")
        },
        code_sha256=h.hexdigest(),
        contract_sha256=file_hash(ROOT / "configs/contracts/planning_v3.json"),
        manifest_sha256=file_hash(manifest),
        teacher=FrenetTeacher.version,
    )


def selected_scenes(manifest, split):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    if data["split_version"] != "split_v2":
        raise ValueError("split version mismatch")
    groups = {}
    selected = []
    seen = set()
    for row in data["rows"]:
        group = row["group_id"]
        bucket = (
            int(hashlib.sha256(("split_v2|" + group).encode()).hexdigest(), 16) % 100
        )
        expected = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        if row["split"] != expected:
            raise ValueError("manifest split/hash mismatch")
        if group in groups and groups[group] != expected:
            raise ValueError("cross-split group")
        groups[group] = expected
        if expected != split:
            continue
        if split == "test":
            raise ValueError("test remains held out")
        key = (row["scene_id"], row["provider"])
        if key in seen:
            raise ValueError("duplicate scene/provider")
        seen.add(key)
        path = (manifest.parent / row["cached_proto"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError("candidate outside inventory")
        if file_hash(path) != row["raw_sha256"]:
            raise ValueError("candidate checksum mismatch")
        selected.append((row, path))
    if not selected:
        raise ValueError("no eligible scenes")
    return selected


def save_collection_boundary(output, replay, report, rng, order, next_index, cycles):
    """Atomically retain two episode-boundary recovery points."""
    path = output / "recovery" / f"boundary-{len(report['episodes']):06d}"
    trace = output / "ticks.jsonl"
    state = dict(
        phase="TEACHER_COLLECTION",
        report=report,
        rng=rng.bit_generator.state,
        order=order,
        next_index=next_index,
        cycles=cycles,
        trace_bytes=trace.stat().st_size if trace.exists() else 0,
    )
    save_checkpoint(path, {}, replay, replay.versions, state, episode_boundary=True)
    write_report(
        output / "latest_recovery.json", dict(path=str(path.relative_to(output)))
    )
    for old in sorted(path.parent.glob("boundary-*"))[:-2]:
        shutil.rmtree(old)


def restore_collection_boundary(output, version):
    pointer = json.loads((output / "latest_recovery.json").read_text())
    path = (output / pointer["path"]).resolve()
    if not path.is_relative_to((output / "recovery").resolve()):
        raise ValueError("invalid recovery path")
    _, replay, state = load_checkpoint(path, {}, version)
    if state["phase"] != "TEACHER_COLLECTION":
        raise ValueError("not a collection recovery point")
    trace = output / "ticks.jsonl"
    committed = state["trace_bytes"]
    if trace.exists():
        if trace.stat().st_size < committed:
            raise ValueError("committed trace is incomplete")
        if trace.stat().st_size > committed:
            # Preserve interrupted work for diagnosis, but keep the main trace
            # aligned with the replay that is actually resumed.
            archive = output / f"uncheckpointed-{time.time_ns()}.jsonl"
            with trace.open("rb") as source, archive.open("wb") as destination:
                source.seek(committed)
                shutil.copyfileobj(source, destination)
            with trace.open("r+b") as stream:
                stream.truncate(committed)
    elif committed:
        raise ValueError("committed trace is missing")
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    return replay, state, rng


def collect_teacher(manifest, output, seed=0, target=20000, smoke=False, resume=False):
    if not 0 < target <= 50000:
        raise ValueError("target must fit replay capacity")
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("fresh output required; no accidental overwrite")
    if resume and smoke:
        raise ValueError("smoke batch boundaries are not resumable episode checkpoints")
    output.mkdir(parents=True, exist_ok=resume)
    version = versions(manifest)
    if resume and (output / "report.json").exists():
        finished = json.loads((output / "report.json").read_text())
        if finished.get("status") == "COMPLETE" and finished.get("qualified"):
            if (
                finished["versions"] != version
                or finished["target"] != target
                or finished["seed"] != seed
                or file_hash(output / "replay.npz") != finished["replay_sha256"]
            ):
                raise ValueError("completed collection provenance mismatch")
            return finished
    replay = Replay(50000, seed=seed, versions=version)
    candidates = selected_scenes(manifest, "train")
    rng = np.random.default_rng(seed)
    report = dict(
        status="RUNNING",
        scope="SMOKE" if smoke else "TEACHER_COLLECTION",
        versions=version,
        seed=seed,
        target=target,
        episodes=[],
        valid_transitions=0,
        accepted_teacher=0,
        replay_array_bytes=replay.nbytes,
        group_status="scenario_fallback_overlap_unverified",
    )
    started = time.perf_counter()
    teacher = FrenetTeacher()
    cycles = 0
    order = None
    next_index = 0
    elapsed_before = 0.0
    if resume:
        replay, state, rng = restore_collection_boundary(output, version)
        report = state["report"]
        if report["seed"] != seed or report["target"] != target:
            raise ValueError("resume seed/target mismatch")
        cycles, order, next_index = state["cycles"], state["order"], state["next_index"]
        elapsed_before = report.get("elapsed_s", 0.0)
        report.update(status="RUNNING", qualified=False)
    checkpoint_at = replay.size + 1000
    write_report(output / "report.json", report)
    try:
        while replay.size < target:
            cycle_start = replay.size
            if order is None:
                order = rng.permutation(len(candidates)).tolist()
                next_index = 0
            from_cycle_start = next_index == 0
            for position in range(next_index, len(order)):
                index = order[position]
                row, path = candidates[index]
                episode = dict(
                    scene_id=row["scene_id"],
                    provider=row["provider"],
                    ticks=0,
                    accepted=0,
                    return_training=0.0,
                    outcome=None,
                )
                report["episodes"].append(episode)
                e = SACPlanningEnvironment()
                try:
                    boundary = e.reset(
                        parse_scenario(path.read_bytes()),
                        seed=seed,
                        provider_spec=row["provider"],
                    )
                    while True:
                        collected = collect_step(
                            e, boundary, teacher.propose, replay, teacher=True
                        )
                        result, boundary = collected.result, collected.boundary
                        episode["ticks"] += 1
                        episode["accepted"] += int(result.check_result.valid)
                        episode["return_training"] += collected.reward.training
                        with (output / "ticks.jsonl").open("a") as stream:
                            record = dict(
                                scene_id=row["scene_id"],
                                provider=row["provider"],
                                episode_uid=e.episode_uid,
                                tick=e.tick,
                                check=asdict(result.check_result),
                                source=result.source.value,
                                pre_token=to_wire(result.pre_token),
                                next_token=to_wire(boundary.token),
                                executed_command_snapshot=to_wire(
                                    result.executed_command_snapshot
                                ),
                                raw_proposal=to_wire(result.raw_proposal),
                                actual_control=to_wire(result.actual_control),
                                tracking_error=(
                                    asdict(result.tracking_error)
                                    if result.tracking_error
                                    else None
                                ),
                                event_flags=result.event_flags,
                                terminated=result.terminated,
                                truncated=result.truncated,
                                sample_valid=result.sample_valid,
                                actual_state=(
                                    dict(
                                        world_xy=e.ego.position.tolist(),
                                        velocity_world=e.ego.velocity.tolist(),
                                        yaw=e.ego.yaw,
                                    )
                                    if result.sample_valid
                                    else None
                                ),
                                waymax_metrics=dict(
                                    valid=e.metrics_valid,
                                    offroad=e.waymax_offroad,
                                    overlap=e.waymax_overlap,
                                ),
                                clipped_features=e.clipped_features,
                                reward=asdict(collected.reward),
                                terminal_reason=(
                                    result.terminal_reason.value
                                    if result.terminal_reason
                                    else None
                                ),
                                teacher=collected.teacher_diagnostics,
                            )
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                        if result.terminated or result.truncated:
                            episode["outcome"] = result.terminal_reason.value
                            break
                        if smoke and episode["ticks"] >= 10:
                            episode["outcome"] = "SMOKE_BATCH_BOUNDARY"
                            break
                except Exception as exc:
                    episode["outcome"] = "COLLECTION_ERROR"
                    episode["error"] = f"{type(exc).__name__}: {exc}"
                    # No invented terminal transition or replacement action.
                report.update(
                    valid_transitions=replay.size,
                    accepted_teacher=int(
                        replay.arrays["teacher_accepted"][: replay.size].sum()
                    ),
                    elapsed_s=elapsed_before + time.perf_counter() - started,
                )
                write_report(output / "report.json", report)
                print(
                    json.dumps(
                        dict(
                            episode=episode,
                            valid_transitions=replay.size,
                            accepted_teacher=report["accepted_teacher"],
                        )
                    ),
                    flush=True,
                )
                next_index = position + 1
                if not smoke and (
                    not (output / "latest_recovery.json").exists()
                    or replay.size >= checkpoint_at
                    or replay.size >= target
                ):
                    save_collection_boundary(
                        output, replay, report, rng, order, next_index, cycles
                    )
                    checkpoint_at = replay.size + 1000
                if replay.size >= target:
                    break
                if smoke and len(report["episodes"]) >= 4:
                    break
            cycles += 1
            order = None
            next_index = 0
            if smoke or (replay.size == cycle_start and from_cycle_start):
                break
            if report["accepted_teacher"] == 0:
                raise RuntimeError("no accepted teacher paths after a full train pass")
        replay.save(output / "replay.npz")
        report.update(
            status="COMPLETE" if replay.size >= target else "INSUFFICIENT_DATA",
            qualified=not smoke
            and replay.size >= 20000
            and report["accepted_teacher"] > 0,
            cycles=cycles,
            replay_sha256=file_hash(output / "replay.npz"),
        )
        if versions(manifest) != version:
            raise RuntimeError("runtime changed during collection")
    except BaseException as exc:
        report.update(
            status="ERROR", qualified=False, error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        report["elapsed_s"] = elapsed_before + time.perf_counter() - started
        write_report(output / "report.json", report)
    return report


def train_bc(manifest, data, output, seed=0, updates=5000, smoke=False):
    version = versions(manifest)
    data = Path(data)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((data / "report.json").read_text())
    if (
        report["versions"] != version
        or file_hash(data / "replay.npz") != report["replay_sha256"]
    ):
        raise ValueError("collection provenance mismatch")
    if not smoke and not report.get("qualified"):
        raise ValueError("BC requires qualified 20000-transition collection")
    if not smoke and updates < 5000:
        raise ValueError("contract requires 5000 BC minibatches")
    replay = Replay.load(data / "replay.npz", version)
    # Each BC run uses its own sampling RNG and learner seed; data is unchanged.
    replay.rng = np.random.default_rng(seed)
    state = initialize(tensor_observation(CoreObservation.empty()), seed)
    started = time.perf_counter()
    losses = []
    for step in range(updates):
        state, loss = bc_update(state, replay.sample(256, teacher_only=True))
        losses.append(float(loss))
        if (step + 1) % 100 == 0:
            print(
                json.dumps(
                    dict(
                        bc_updates=step + 1,
                        loss=losses[-1],
                        elapsed_s=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
    run = dict(
        phase="BC",
        qualified=not smoke,
        seed=seed,
        bc_updates=updates,
        collection_sha256=report["replay_sha256"],
        first_loss=losses[0],
        last_loss=losses[-1],
        elapsed_s=time.perf_counter() - started,
        policy_performance_validated=False,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    )
    import jax

    run["devices"] = [str(device) for device in jax.devices()]
    run["device_memory_stats"] = [device.memory_stats() for device in jax.devices()]
    save_checkpoint(output, state, replay, version, run, episode_boundary=True)
    write_report(output / "bc_report.json", run)
    return run
