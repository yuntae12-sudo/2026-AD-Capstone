"""Explicit experimental amendment: no demonstrations or pretrained weights."""

import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "configs/experiments/pure_sac_v1.json"
)
WARMUP = 5000


def protocol_metadata():
    data = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(data)
    if protocol["warmup_transitions"] != WARMUP:
        raise ValueError("pure SAC warmup protocol mismatch")
    return dict(id=protocol["id"], sha256=hashlib.sha256(data).hexdigest())


def expected_updates(transitions):
    return max(0, transitions - WARMUP)


def pure_update(state, replay, transitions):
    """First 5000 actual transitions collect only; never backfill their updates."""
    from src.training.sac.learner import update

    if int(state.bc_updates) != 0 or int(state.updates) != expected_updates(
        transitions - 1
    ):
        raise ValueError("pure SAC learner counter mismatch before update")
    if replay.size != min(transitions, replay.capacity):
        raise ValueError("pure SAC learner replay count mismatch")
    if transitions <= WARMUP:
        return state, None
    return update(state, replay.sample(256), replay.size)


def validate_pure_state(state, replay, transitions):
    if int(state.bc_updates) != 0:
        raise ValueError("pure SAC forbids BC weights")
    if int(state.updates) != expected_updates(transitions):
        raise ValueError("pure SAC update/warmup counter mismatch")
    if replay.size != min(transitions, replay.capacity):
        raise ValueError("pure SAC replay must contain only its own transitions")
    if np.any(replay.arrays["teacher_accepted"][: replay.size]):
        raise ValueError("pure SAC forbids teacher replay")


def require_synthetic_report(report, version, runner_sha256):
    # This gate validates exploration around prescribed synthetic paths, not
    # whether the untrained actor generates good paths on actual roads.
    from scripts.validate_sac_exploration import CASES

    expected = {
        (f"v{v}_a{a}_k{k}", sigma) for v, a, k in CASES for sigma in (0.02, 0.03)
    }
    rows = report.get("cases", [])
    if (
        report.get("versions") != version
        or report.get("runner_sha256") != runner_sha256
        or report.get("scope") != "SYNTHETIC_KINEMATICS"
        or report.get("status") != "COMPLETE"
        or report.get("qualified") is not True
        or report.get("passed") is not True
        or report.get("requested_samples") != 4096
        or len(rows) != 18
        or {(r["case"], r["sigma"]) for r in rows} != expected
        or any(
            r.get("mean_valid") is not True
            or r["samples"] != 4096
            or not 3892 <= r["accepted"] <= 4096
            for r in rows
        )
    ):
        raise ValueError("pure SAC requires the complete matching synthetic gate")


def validate_pure_evaluation_checkpoint(state, run):
    report = run.get("report", {})
    if (
        run.get("phase") != "ONLINE_SAC"
        or int(state.bc_updates) != 0
        or int(state.updates) < 10000
        or report.get("training_mode") != "pure-sac"
        or report.get("protocol") != protocol_metadata()
        or report.get("initial_replay_size") != 0
        or report.get("bc_checkpoint_sha256") is not None
        or int(state.updates)
        != expected_updates(report.get("valid_online_transitions", 0))
    ):
        raise ValueError("not a completed pure SAC checkpoint")


def validate_rule_baseline(report, version, expected_pairs):
    """Reuse an unchanged, complete Frenet evaluation without rerunning Waymax."""
    from scripts.sac_experiment_runtime import summarize_episodes

    episodes = report.get("episodes", [])
    pairs = [(e["scene_id"], e["provider"]) for e in episodes]
    if (
        report.get("scope") != "VALIDATION_PERFORMANCE"
        or report.get("status") != "COMPLETE"
        or report.get("qualified") is not True
        or report.get("policy") != "teacher"
        or report.get("seed") != 0
        or report.get("versions") != version
        or len(pairs) != len(set(pairs))
        or set(pairs) != set(expected_pairs)
        or any(
            e.get("outcome") in (None, "DATA_CORRUPTION", "SMOKE_BATCH_BOUNDARY")
            for e in episodes
        )
        or report.get("by_behavior") != summarize_episodes(episodes)
    ):
        raise ValueError("cached rule-based evaluation coverage/runtime mismatch")
