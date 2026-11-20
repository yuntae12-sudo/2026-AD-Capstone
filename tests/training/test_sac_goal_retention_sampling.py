from types import SimpleNamespace
import numpy as np
import pytest
from scripts.sac_goal_retention_sampling import draw_indices, goal_terminal_indices, sample_recent


def test_goal_band_requires_a_real_terminal_not_budget_bootstrap():
    reward = np.array([.05, .12, .1, .02, -.05, -20., .049999, .120001])
    bootstrap = np.array([False, False, True, False, False, False, False, False])
    np.testing.assert_array_equal(goal_terminal_indices(8, bootstrap, reward), [0, 1])


def test_never_reads_future_capacity_rows_and_retains_recent_distribution():
    reward = np.zeros(1000); bootstrap = np.ones(1000, bool)
    reward[[2, 998]] = .1; bootstrap[[2, 998]] = False
    indices = draw_indices(np.random.default_rng(0), 700, bootstrap, reward)
    assert len(indices) == 256 and np.all((indices >= 0) & (indices < 700))
    assert (indices[128:248] >= 444).all()
    np.testing.assert_array_equal(indices[248:], np.full(8, 2))
    assert (indices[:128] < 444).any()


def test_no_goal_does_not_invent_success_and_uses_uniform_fallback():
    reward = np.zeros(700); bootstrap = np.ones(700, bool)
    indices = draw_indices(np.random.default_rng(4), 700, bootstrap, reward)
    assert len(indices) == 256 and ((0 <= indices) & (indices < 700)).all()
    assert (indices[248:] < 444).any()


@pytest.mark.parametrize('size,bootstrap,reward', [
    (0, np.ones(1, bool), np.zeros(1)),
    (2, np.ones(1, bool), np.zeros(2)),
    (1, np.array([1]), np.zeros(1)),
    (1, np.ones(1, bool), np.array([np.nan])),
])
def test_corrupt_prefix_rejected(size, bootstrap, reward):
    with pytest.raises(ValueError):
        goal_terminal_indices(size, bootstrap, reward)


def test_teacher_experience_is_rejected_before_sampling():
    replay = SimpleNamespace(size=1, cursor=1, capacity=2, arrays={'teacher_accepted': np.array([True])})
    with pytest.raises(ValueError, match='teacher0'):
        sample_recent(replay)


@pytest.mark.parametrize('ending,expected', [('ROUTE_GOAL', [0]), ('COLLISION', []), ('RECORD_END', []), ('TASK_LOST', [])])
def test_pool_labels_match_actual_reward_contract(ending, expected):
    from src.contracts.planning import Maneuver, Source, TerminalReason
    from src.training.sac.reward import calculate_reward
    command = SimpleNamespace(maneuver=Maneuver.KEEP, desired_speed_mps=10.)
    result = calculate_reward(command, delta_s=0., speed=0., speed_limit=10., gap=None,
        ttc=float('inf'), acceleration=0., jerk=None, lateral_acceleration=0.,
        tracking_error=None, raw_rejected=False, source=Source.NEW,
        terminal_reason=TerminalReason(ending))
    np.testing.assert_array_equal(goal_terminal_indices(1, np.array([False]), np.array([result.training])), expected)
