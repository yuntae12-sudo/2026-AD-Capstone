"""Terminal retention preserves real replay contents and sampling lineage."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.sac_terminal_retention_sampling import (
    FAILURE_REASONS, draw_indices, sample_recent, terminal_pools,
)
from scripts.sac_diagnostic_sampling import indices_hash


@pytest.mark.parametrize("ending", [
    "ROUTE_GOAL", "COLLISION", "OFFROAD", "CORRIDOR_VIOLATION",
    "MAP_COVERAGE_EXIT", "RECORD_END", "TIME_LIMIT", "TASK_LOST",
])
def test_actual_reward_distinguishes_failure_goal_and_other_endings(ending):
    from src.contracts.planning import Maneuver, Source, TerminalReason
    from src.training.sac.reward import calculate_reward
    command = SimpleNamespace(maneuver=Maneuver.KEEP, desired_speed_mps=10.)
    result = calculate_reward(
        command, delta_s=0., speed=0., speed_limit=10., gap=None,
        ttc=float("inf"), acceleration=0., jerk=None, lateral_acceleration=0.,
        tracking_error=None, raw_rejected=False, source=Source.NEW,
        terminal_reason=TerminalReason(ending),
    )
    assert result.sample_valid
    goal, failure = terminal_pools(1, np.array([False]), np.array([result.training]))
    np.testing.assert_array_equal(goal, [0] if ending == "ROUTE_GOAL" else [])
    np.testing.assert_array_equal(failure, [0] if ending in FAILURE_REASONS else [])
    # Neither event-looking rewards nor a diagnostic budget cut make a
    # continuing transition a terminal. Bootstrap is mandatory evidence.
    goal, failure = terminal_pools(1, np.array([True]), np.array([result.training]))
    assert len(goal) == len(failure) == 0


def test_slots_use_only_existing_eligible_prefix_rows():
    reward = np.zeros(1000); bootstrap = np.ones(1000, bool)
    reward[[2, 998]] = .1; reward[[3, 999]] = -20.
    bootstrap[[2, 3, 998, 999]] = False
    indices = draw_indices(np.random.default_rng(0), 700, bootstrap, reward)
    assert len(indices) == 256 and np.all((indices >= 0) & (indices < 700))
    assert (indices[:128] < 444).any()
    assert (indices[128:240] >= 444).all()
    np.testing.assert_array_equal(indices[240:248], np.full(8, 2))
    np.testing.assert_array_equal(indices[248:], np.full(8, 3))


@pytest.mark.parametrize("goal_present,failure_present", [(False, False), (True, False), (False, True)])
def test_each_empty_pool_falls_back_independently_without_fabricating_rows(goal_present, failure_present):
    reward = np.zeros(700); bootstrap = np.ones(700, bool)
    if goal_present:
        reward[2] = .1; bootstrap[2] = False
    if failure_present:
        reward[3] = -20.; bootstrap[3] = False
    rng = np.random.default_rng(4)
    expected = np.concatenate((
        rng.choice(np.arange(700), 128), rng.choice(np.arange(444, 700), 112),
        rng.choice([2] if goal_present else np.arange(700), 8),
        rng.choice([3] if failure_present else np.arange(700), 8),
    ))
    actual_rng = np.random.default_rng(4)
    np.testing.assert_array_equal(draw_indices(actual_rng, 700, bootstrap, reward), expected)
    assert rng.bit_generator.state == actual_rng.bit_generator.state


@pytest.mark.parametrize("size,bootstrap,reward", [
    (0, np.ones(1, bool), np.zeros(1)),
    (2, np.ones(1, bool), np.zeros(2)),
    (1, np.array([1]), np.zeros(1)),
    (1, np.ones(1, bool), np.array([np.nan])),
])
def test_corrupt_input_rejected_without_advancing_rng(size, bootstrap, reward):
    rng = np.random.default_rng(1); before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        draw_indices(rng, size, bootstrap, reward)
    assert before == rng.bit_generator.state


def test_real_replay_gather_and_saved_rng_preserve_actions_masks_and_rewards(tmp_path):
    from src.training.sac.replay import Replay, unpack_observation
    replay = Replay(capacity=310, seed=7)
    for value in replay.arrays.values():
        value[...] = False if value.dtype == bool else 0
    for i in range(300):
        for key in ("obs", "next_obs", "raw_u", "tick"):
            replay.arrays[key][i] = i
        for key in ("fields", "next_fields", "items", "next_items"):
            replay.arrays[key][i] = bool(i % 2)
    replay.arrays["reward"][[2, 3]] = [.1, -20.]
    replay.size = replay.cursor = 300
    before = {k: v.copy() for k, v in replay.arrays.items()}
    expected = draw_indices(np.random.default_rng(7), 300, before["bootstrap"], before["reward"])
    batch, digest = sample_recent(replay)
    assert digest == indices_hash(expected)
    for key in ("raw_u", "reward", "bootstrap", "discount", "delta_ticks", "terminated", "truncated"):
        np.testing.assert_array_equal(batch[key], before[key][expected])
    for prefix, name in (("", "observation"), ("next_", "next_observation")):
        observation = unpack_observation(*(before[prefix + key][expected] for key in ("obs", "fields", "items")))
        for category in observation:
            for key, values in observation[category].items():
                np.testing.assert_array_equal(batch[name][category][key], values)
    assert batch["raw_u"].dtype == np.float64
    for key in before:
        assert before[key].tobytes() == replay.arrays[key].tobytes()
    path = tmp_path / "replay.npz"; replay.save(path)
    restored = Replay.load(path, {})
    a, ah = sample_recent(replay); b, bh = sample_recent(restored)
    assert ah == bh and replay.rng.bit_generator.state == restored.rng.bit_generator.state
    np.testing.assert_array_equal(a["raw_u"], b["raw_u"])


@pytest.mark.parametrize("size,cursor,teacher", [(0, 0, False), (2, 0, False), (1, 0, False), (1, 1, True)])
def test_invalid_ring_or_teacher_never_consumes_rng(size, cursor, teacher):
    replay = SimpleNamespace(size=size, cursor=cursor, capacity=2,
        arrays={"teacher_accepted": np.array([teacher, False])}, rng=np.random.default_rng(7))
    before = copy.deepcopy(replay.rng.bit_generator.state)
    with pytest.raises(ValueError):
        sample_recent(replay)
    assert before == replay.rng.bit_generator.state
