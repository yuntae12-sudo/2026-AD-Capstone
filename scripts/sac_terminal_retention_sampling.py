"""Opt-in replay comparison retaining actual own goal and failure terminals.

128 uniform + 112 recent + 8 goal + 8 severe-failure rows, with replacement.
Each empty terminal pool falls back to uniform. No rewards are synthesized and
no importance weights are used; actor and critic state distributions both change.
Historical samplers and the production replay implementation stay unchanged.
"""
import hashlib
from pathlib import Path

import numpy as np

from scripts.sac_diagnostic_sampling import indices_hash
from scripts.sac_goal_retention_sampling import goal_terminal_indices

MODE = "own_goal8_failure8"
FAILURE_REASONS = frozenset(("COLLISION", "OFFROAD", "CORRIDOR_VIOLATION", "MAP_COVERAGE_EXIT"))


def sampling_spec(mode=MODE):
    if mode != MODE:
        raise ValueError("this comparison supports only own_goal8_failure8")
    return dict(
        mode=MODE, batch_size=256, uniform_draws=128, recent_draws=112,
        recent_window=256, goal_terminal_draws=8, failure_terminal_draws=8,
        replacement=True, empty_goal_pool="uniform", empty_failure_pool="uniform",
        goal_reward_interval=[.05, .12], failure_reward_interval=[-20.05, -19.98],
        terminal_requires_no_bootstrap=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        goal_classifier_sha256=hashlib.sha256(
            Path(__file__).with_name("sac_goal_retention_sampling.py").read_bytes()
        ).hexdigest(),
    )


def terminal_pools(size, bootstrap, reward):
    # Reuse its finite-prefix, size and boolean-bootstrap validation.
    goals = goal_terminal_indices(size, bootstrap, reward)
    bootstrap, reward = np.asarray(bootstrap), np.asarray(reward)
    # v1.2 actual failure event -2000, clipped step [-5,2], training /100.
    # The run audit independently checks these rows against actual outcome labels.
    failures = np.flatnonzero(
        ~bootstrap[:size] & (reward[:size] >= -20.05) & (reward[:size] <= -19.98)
    )
    return goals, failures


def draw_indices(rng, size, bootstrap, reward):
    goals, failures = terminal_pools(size, bootstrap, reward)
    all_rows = np.arange(size)
    return np.concatenate((
        rng.choice(all_rows, 128, replace=True),
        rng.choice(np.arange(max(0, size - 256), size), 112, replace=True),
        rng.choice(goals if len(goals) else all_rows, 8, replace=True),
        rng.choice(failures if len(failures) else all_rows, 8, replace=True),
    ))


def sample_recent(replay):
    if replay.size != replay.cursor or not 0 < replay.size < replay.capacity:
        raise ValueError("retention diagnostic requires nonempty replay without wrap")
    if replay.arrays["teacher_accepted"][:replay.size].any():
        raise ValueError("terminal retention requires teacher0")
    from src.training.sac.replay import unpack_observation
    indices = draw_indices(replay.rng, replay.size, replay.arrays["bootstrap"], replay.arrays["reward"])
    result = {k: v[indices] for k, v in replay.arrays.items() if k not in ("episode_uid", "scene_id")}
    result["observation"] = unpack_observation(result.pop("obs"), result.pop("fields"), result.pop("items"))
    result["next_observation"] = unpack_observation(result.pop("next_obs"), result.pop("next_fields"), result.pop("next_items"))
    return result, indices_hash(indices)
