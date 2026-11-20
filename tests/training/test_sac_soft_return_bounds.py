import numpy as np
import pytest

from scripts.sac_soft_return_bounds import conditional_return_bounds, entropy_bounds


def test_contract_sampler_entropy_is_inside_bounds_at_sigma_extremes():
    import jax.numpy as jnp
    from src.training.sac.networks import sample_action
    rng = np.random.default_rng(19)
    epsilon = rng.normal(size=(4, 30, 2))
    low, high = entropy_bounds(epsilon)
    # Unequal axes, both saturation endpoints, and state-dependent g.
    rho = jnp.asarray([[-1000., -1000.], [1000., 1000.], [-1000., 1000.], [-.7, 2.3]])
    _, ell, _ = sample_action(jnp.zeros((4, 30, 2)), rho, jnp.asarray([0., .1, 2., 10.]), None, epsilon=jnp.asarray(epsilon))
    ell = np.asarray(ell)
    assert np.all(low <= ell) and np.all(ell <= high)
    assert abs(ell[0]-high[0]) < 2e-10 and abs(ell[1]-low[1]) < 2e-10


def test_conditional_action_entropy_is_excluded_and_terminal_is_exact():
    reward, low, high = conditional_return_bounds([1., -20.], [-999., -7.], [999., -3.], [True, False], .1)
    np.testing.assert_allclose(reward, [-18.8, -20.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(low, [1.+.99*(-20.+.3), -20.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(high, [1.+.99*(-20.+.7), -20.], rtol=0, atol=1e-12)


def test_all_realized_suffix_returns_fit_bounds_using_independent_discounted_sums():
    rng = np.random.default_rng(23)
    eps = rng.normal(size=(57, 30, 2))
    sigma = rng.uniform(.005, .03, size=(57, 2))
    ell = 30.-.5*np.sum(eps**2, axis=(1,2))-30.*np.log(sigma/.02).sum(axis=1)
    ell_low, ell_high = entropy_bounds(eps)
    rewards = rng.normal(size=57); rewards[-1] = -20.
    alpha = .000736
    reward_only, low, high = conditional_return_bounds(rewards, ell_low, ell_high, [True]*56+[False], alpha)
    for i in range(57):
        direct_reward = sum(.99**(j-i)*rewards[j] for j in range(i,57))
        direct = direct_reward-alpha*sum(.99**(j-i)*ell[j] for j in range(i+1,57))
        assert abs(reward_only[i]-direct_reward)<1e-10
        assert low[i]-1e-10 <= direct <= high[i]+1e-10
    np.testing.assert_allclose(high-low, [alpha*60*np.log(6)*sum(.99**j for j in range(1,57-i)) for i in range(57)], rtol=0, atol=1e-10)


def test_zero_alpha_removes_entropy_uncertainty():
    reward, low, high = conditional_return_bounds([1.,2.,3.], [-5.]*3, [7.]*3, [True,True,False], 0.)
    np.testing.assert_array_equal(reward,low);np.testing.assert_array_equal(reward,high)


@pytest.mark.parametrize('bootstrap', [[True,True],[False,False]])
def test_budget_cutoff_or_internal_terminal_is_rejected(bootstrap):
    with pytest.raises(ValueError):conditional_return_bounds([1.,2.], [-3.,-3.], [4.,4.], bootstrap, .1)


@pytest.mark.parametrize('alpha',[-.1, float('nan')])
def test_invalid_temperature_is_rejected(alpha):
    with pytest.raises(ValueError):conditional_return_bounds([1.], [-3.], [4.], [False], alpha)


def test_reversed_entropy_bounds_are_rejected():
    with pytest.raises(ValueError):conditional_return_bounds([1.], [4.], [-3.], [False], .1)
