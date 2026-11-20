"""Opt-in recent replay weighting for bounded no-wrap self-SAC diagnostics.

This changes the distribution of training examples, not the Gaussian behavior
policy or SAC loss formulas. Production Replay.sample remains uniform.
"""

import hashlib
from pathlib import Path

import numpy as np

MODES = ("uniform", "half_recent256")


def sampling_spec(mode="uniform"):
    if mode not in MODES:
        raise ValueError("unknown diagnostic replay sampling mode")
    if mode == "uniform":
        return {"mode": mode}
    return dict(mode=mode, batch_size=256, uniform_draws=128, recent_draws=128,
                recent_window=256, replacement=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def report_sampling_mode(report):
    spec = report.get("replay_sampling", {"mode": "uniform"})
    mode = spec.get("mode")
    if spec != sampling_spec(mode):
        raise ValueError("diagnostic replay sampling specification or implementation differs")
    return mode


def draw_indices(rng, size, mode="uniform"):
    if mode not in MODES or size <= 0:
        raise ValueError("nonempty replay and known sampling mode required")
    if mode == "uniform":
        return rng.choice(np.arange(size), 256, replace=True)
    return np.concatenate((rng.choice(np.arange(size), 128, replace=True),
                           rng.choice(np.arange(max(0, size - 256), size),
                                      128, replace=True)))


def indices_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def sample_recent(replay):
    """Gather original stored rows, with auditable indices and no extra update."""
    from src.training.sac.replay import unpack_observation

    if replay.size != replay.cursor or not 0 < replay.size < replay.capacity:
        raise ValueError("recent diagnostic requires nonempty replay without wrap")
    indices = draw_indices(replay.rng, replay.size, "half_recent256")
    result = {k: v[indices] for k, v in replay.arrays.items()
              if k not in ("episode_uid", "scene_id")}
    result["observation"] = unpack_observation(
        result.pop("obs"), result.pop("fields"), result.pop("items"))
    result["next_observation"] = unpack_observation(
        result.pop("next_obs"), result.pop("next_fields"), result.pop("next_items"))
    return result, indices_hash(indices)
