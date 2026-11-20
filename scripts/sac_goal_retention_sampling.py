"""Opt-in retention of real own-goal terminal experience in a SAC batch.

128 uniform + 120 recent + 8 own-goal terminal draws, with replacement.
Empty goal pools fall back to uniform. This changes replay probabilities, not
rewards, loss formulas, Gaussian sampling, or the one-update-per-transition rule.
The old sampler stays immutable for historical experiment audits.
"""
import hashlib
from pathlib import Path
import numpy as np
from scripts.sac_diagnostic_sampling import indices_hash

MODE = "own_goal_terminal8"


def sampling_spec(mode=MODE):
    if mode != MODE:
        raise ValueError("this comparison supports only own_goal_terminal8")
    return dict(mode=MODE, batch_size=256, uniform_draws=128, recent_draws=120,
                recent_window=256, goal_terminal_draws=8, replacement=True,
                empty_goal_pool="uniform", goal_reward_interval=[.05, .12],
                goal_requires_no_bootstrap=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def goal_terminal_indices(size, bootstrap, reward):
    bootstrap, reward = np.asarray(bootstrap), np.asarray(reward)
    if (type(size) is not int or size <= 0 or bootstrap.ndim != 1 or reward.ndim != 1
            or len(bootstrap) < size or len(reward) < size or bootstrap.dtype != np.bool_
            or not np.isfinite(reward[:size]).all()):
        raise ValueError("requires nonempty finite replay prefix and boolean bootstrap")
    # v1.2: goal event +10, clipped step [-5,2], training reward /100.
    # Other endings have event 0, -5 or -2000; their rewards cannot be in this band.
    return np.flatnonzero(~bootstrap[:size] & (reward[:size] >= .05) & (reward[:size] <= .12))


def draw_indices(rng, size, bootstrap, reward):
    goal = goal_terminal_indices(size, bootstrap, reward)
    all_rows = np.arange(size)
    return np.concatenate((rng.choice(all_rows, 128, replace=True),
                           rng.choice(np.arange(max(0, size-256), size), 120, replace=True),
                           rng.choice(goal if len(goal) else all_rows, 8, replace=True)))


def sample_recent(replay):
    if replay.size != replay.cursor or not 0 < replay.size < replay.capacity:
        raise ValueError("retention diagnostic requires nonempty replay without wrap")
    if replay.arrays["teacher_accepted"][:replay.size].any():
        raise ValueError("own-goal retention requires teacher0")
    from src.training.sac.replay import unpack_observation
    indices = draw_indices(replay.rng, replay.size, replay.arrays["bootstrap"], replay.arrays["reward"])
    result = {k: v[indices] for k, v in replay.arrays.items() if k not in ("episode_uid", "scene_id")}
    result["observation"] = unpack_observation(result.pop("obs"), result.pop("fields"), result.pop("items"))
    result["next_observation"] = unpack_observation(result.pop("next_obs"), result.pop("next_fields"), result.pop("next_items"))
    return result, indices_hash(indices)
