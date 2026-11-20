"""The path decomposition must telescope without a terminal bootstrap."""
import numpy as np
import pytest

from scripts.sac_bellman_trace import decompose


def test_terminal_has_no_next_value_or_entropy():
    r = decompose([-20.], [0.], [-3.], [100.], [False], .01)
    assert r['summary']['initial_gap'] == 17.
    assert r['summary']['terminal_contribution'] == 17.
    assert r['summary']['discounted_target_lag_sum'] == 0.
    assert r['rows'][0]['sampled_target'] == -20.


def test_exact_soft_returns_give_zero_online_residual_with_entropy():
    q1 = -20.
    q0 = 1. + .99 * (q1 - .01 * -2.)
    r = decompose([1., -20.], [0., -2.], [q0, q1], [q0, q1], [True, False], .01)
    assert r['summary']['initial_gap'] == 0.
    np.testing.assert_allclose([v['online_bellman_residual'] for v in r['rows']], 0, atol=1e-12)


def test_target_lag_and_td_residual_partition_same_gap():
    r = decompose([0., 0., -20.], [0., -1., 2.], [-1., -4., -10.],
                  [-.5, -2., -7.], [True, True, False], .01)
    s = r['summary']
    expected_return = .99 * .01 + .99**2 * (-20. - .02)
    assert s['soft_return'] == pytest.approx(expected_return)
    assert s['initial_gap'] == pytest.approx(-1. - expected_return)
    assert s['discounted_target_lag_sum'] == pytest.approx(.99*2 + .99**2*3)
    assert s['discounted_residual_sum'] == pytest.approx(
        s['discounted_target_residual_sum'] + s['discounted_target_lag_sum'])
    assert s['terminal_contribution'] == pytest.approx(.99**2*10)


@pytest.mark.parametrize('bootstrap,ell,q', [
    ([True, True], [0., 0.], [1., 1.]),
    ([False, False], [0., 0.], [1., 1.]),
    ([True, False], [1., 0.], [1., 1.]),
    ([True, False], [0., 0.], [1., np.nan]),
])
def test_incomplete_postterminal_or_invalid_trace_is_rejected(bootstrap, ell, q):
    with pytest.raises(ValueError):
        decompose([0., -20.], ell, q, [1., 1.], bootstrap, .01)
