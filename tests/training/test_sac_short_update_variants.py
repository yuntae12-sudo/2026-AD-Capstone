"""Diagnostic options preserve unbounded Gaussian actions and trainable Q."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.training.sac.networks import Actor, Critic, correlation_matrix, sample_action
from src.training.sac.learner import initialize, update_checked
from tests.training.test_sac_learning import observation


def test_linear_mean_gain_keeps_full_rank_unbounded_outputs_and_parameters():
    obs = observation()
    base, scaled = Actor(), Actor(mean_gain=1e-4)
    variables = base.init(jax.random.PRNGKey(7), obs)
    other = scaled.init(jax.random.PRNGKey(7), obs)
    for a, b in zip(
        jax.tree_util.tree_leaves(variables), jax.tree_util.tree_leaves(other)
    ):
        np.testing.assert_array_equal(a, b)
    # An alternating bias remains alternating: no temporal mean smoothing.
    bias = (
        jnp.arange(60, dtype=jnp.float32) * jnp.where(jnp.arange(60) % 2, -1, 1) * 1e6
    )
    variables["params"]["mu"]["bias"] = bias
    mean, rho = scaled.apply(variables, obs)
    np.testing.assert_allclose(mean[0].reshape(-1), np.asarray(bias) * 1e-4, rtol=1e-6)
    assert np.max(np.abs(mean)) > 1
    jacobian = jax.jacfwd(
        lambda b: scaled.apply(
            {
                "params": {
                    **variables["params"],
                    "mu": {**variables["params"]["mu"], "bias": b},
                }
            },
            obs,
        )[0][0].reshape(-1)
    )(bias)
    np.testing.assert_allclose(jacobian, np.eye(60) * 1e-4, rtol=1e-6)
    np.testing.assert_array_equal(rho, base.apply(variables, obs)[1])


def test_zero_q_starts_without_random_bootstrap_value_and_can_learn():
    obs = observation()
    state = initialize(
        obs, critic=Critic(zero_output=True), actor=Actor(mean_gain=1e-4)
    )
    u = jnp.full((2, 30, 2), 0.01)
    for params in (state.q1.params, state.q2.params, state.target1, state.target2):
        np.testing.assert_array_equal(state.q1.apply_fn({"params": params}, obs, u), 0)
    batch = dict(
        observation=obs,
        next_observation=obs,
        raw_u=u,
        reward=np.array([-1.0, -1.0]),
        delta_ticks=np.ones(2),
        bootstrap=np.zeros(2, bool),
    )
    new, metrics = update_checked(state, batch)
    assert float(metrics["q1_grad_norm"]) > 0
    assert float(metrics["q2_grad_norm"]) > 0
    assert int(new.updates) == 1 and int(new.bc_updates) == 0
    q = new.q1.apply_fn({"params": new.q1.params}, obs, u)
    assert np.all(q < 0) and np.all(np.isfinite(q))


def test_zero_mean_gain_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        Actor(mean_gain=0).init(jax.random.PRNGKey(0), observation())


def test_integrated_mean_is_full_rank_unbounded_and_keeps_sampler_density():
    obs = observation()
    actor = Actor(mean_basis="integrated")
    variables = actor.init(jax.random.PRNGKey(3), obs)
    bias = jnp.linspace(-1000, 1000, 60, dtype=jnp.float32)
    variables["params"]["mu"]["bias"] = bias
    mu, rho = actor.apply(variables, obs)
    b = np.asarray(correlation_matrix(), np.float32)
    assert np.linalg.matrix_rank(b.astype(np.float64)) == 30
    np.testing.assert_allclose(
        mu[0], b @ np.asarray(bias).reshape(30, 2), rtol=1e-6, atol=1e-5
    )
    assert mu.dtype == jnp.float32 and np.max(np.abs(mu)) > 1
    key = jax.random.PRNGKey(44)
    u, ell, density = sample_action(mu, rho, jnp.ones(2) * 5, key)
    noise, ell0, density0 = sample_action(jnp.zeros_like(mu), rho, jnp.ones(2) * 5, key)
    np.testing.assert_allclose(
        u - np.asarray(mu, np.float64), noise, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(ell, ell0)
    np.testing.assert_array_equal(density, density0)


def test_integrated_mean_gradient_retains_all_sixty_coordinates():
    obs = observation()
    actor = Actor(mean_basis="integrated")
    params = actor.init(jax.random.PRNGKey(3), obs)["params"]

    def forward(bias):
        changed = {**params, "mu": {**params["mu"], "bias": bias}}
        return actor.apply({"params": changed}, obs)[0][0].reshape(-1)

    jac = jax.jacfwd(forward)(jnp.zeros(60, jnp.float32))
    expected = np.kron(np.asarray(correlation_matrix(), np.float32), np.eye(2))
    np.testing.assert_allclose(jac, expected, rtol=1e-6)
    assert np.linalg.matrix_rank(np.asarray(jac, np.float64)) == 60
