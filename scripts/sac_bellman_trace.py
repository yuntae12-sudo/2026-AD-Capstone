"""Accounting identities on a completed frozen-policy soft-return trajectory.

These are sampled-path residuals, not an estimate of the expected Bellman error
from a single path. They do not modify a learner, replay, reward, or target.
"""

import numpy as np


def decompose(rewards, ell, online, target, bootstrap, alpha):
    rewards, ell, online, target = (
        np.asarray(value, np.float64) for value in (rewards, ell, online, target)
    )
    bootstrap = np.asarray(bootstrap)
    n = rewards.size
    if (not n or any(x.shape != (n,) for x in (rewards, ell, online, target, bootstrap))
            or bootstrap.dtype != bool or not bootstrap[:-1].all() or bootstrap[-1]
            or not all(np.isfinite(x).all() for x in (rewards, ell, online, target))
            or not np.isfinite(alpha) or alpha < 0 or ell[0] != 0):
        raise ValueError("requires a finite complete one-tick path and entropy-free first action")
    discount = .99 ** np.arange(n)
    next_online = np.r_[online[1:] - alpha * ell[1:], 0.]
    next_target = np.r_[target[1:] - alpha * ell[1:], 0.]
    residual = online - rewards - .99 * next_online
    target_residual = online - rewards - .99 * next_target
    target_lag = .99 * (next_target - next_online)
    returns = np.empty(n, np.float64)
    returns[-1] = rewards[-1]
    for i in range(n - 2, -1, -1):
        returns[i] = rewards[i] + .99 * (returns[i + 1] - alpha * ell[i + 1])
    discounted = discount * residual
    td = discount * target_residual
    lag = discount * target_lag
    gap = online[0] - returns[0]
    np.testing.assert_allclose(discounted, td + lag, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(discounted.sum(), gap, atol=1e-10, rtol=1e-12)
    return dict(
        rows=[dict(index=i, q=float(online[i]), target_q=float(target[i]),
                   observed_soft_return=float(returns[i]), q_minus_observed_return=float(online[i]-returns[i]),
                   sampled_target=float(rewards[i] + .99 * next_target[i]),
                   online_bellman_residual=float(residual[i]), target_bellman_residual=float(target_residual[i]),
                   target_lag=float(target_lag[i]), discounted_residual=float(discounted[i]),
                   discounted_target_residual=float(td[i]), discounted_target_lag=float(lag[i])) for i in range(n)],
        summary=dict(ticks=n, q0=float(online[0]), soft_return=float(returns[0]), initial_gap=float(gap),
                     discounted_residual_sum=float(discounted.sum()),
                     discounted_target_residual_sum=float(td.sum()), discounted_target_lag_sum=float(lag.sum()),
                     terminal_contribution=float(discounted[-1]),
                     nonterminal_contribution=float(discounted[:-1].sum())),
    )
