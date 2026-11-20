"""Bounds on realized conditional soft returns when actor inference differs.

The contract bounds sigma to [0.005, 0.03]. Given the recorded rollout RNG,
these bounds need neither the actor mean nor an inverse of the covariance.
They are not confidence intervals for expected Q or exact reconstructed returns.
"""
import numpy as np

ELL_ROUNDING_MARGIN = 1e-10


def entropy_bounds(epsilon):
    epsilon = np.asarray(epsilon, np.float64)
    if epsilon.shape[-2:] != (30, 2) or not np.isfinite(epsilon).all():
        raise ValueError("requires finite standard-normal noise with shape [...,30,2]")
    base = 30.0 - .5 * np.sum(epsilon ** 2, axis=(-2, -1))
    lower = base - 60.0 * np.log(.03 / .02) - ELL_ROUNDING_MARGIN
    upper = base - 60.0 * np.log(.005 / .02) + ELL_ROUNDING_MARGIN
    return lower, upper


def conditional_return_bounds(rewards, ell_lower, ell_upper, bootstrap, alpha):
    rewards, lower, upper = (np.asarray(x, np.float64) for x in (rewards, ell_lower, ell_upper))
    bootstrap = np.asarray(bootstrap)
    n = rewards.size
    if (rewards.ndim != 1 or not n or lower.shape != (n,) or upper.shape != (n,)
            or bootstrap.shape != (n,) or bootstrap.dtype != bool
            or not bootstrap[:-1].all() or bootstrap[-1]
            or not all(np.isfinite(x).all() for x in (rewards, lower, upper))
            or np.any(lower > upper) or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("requires a complete finite one-tick trajectory and ordered entropy bounds")
    reward_only = np.empty(n, np.float64)
    low = np.empty(n, np.float64)
    high = np.empty(n, np.float64)
    reward_only[-1] = low[-1] = high[-1] = rewards[-1]
    for i in range(n-2, -1, -1):
        reward_only[i] = rewards[i] + .99 * reward_only[i+1]
        low[i] = rewards[i] + .99 * (low[i+1] - alpha * upper[i+1])
        high[i] = rewards[i] + .99 * (high[i+1] - alpha * lower[i+1])
    return reward_only, low, high
