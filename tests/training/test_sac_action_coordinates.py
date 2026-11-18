"""Section-5 critic coordinates preserve physical actions and actor gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.training.sac.networks import (
    Actor,
    Critic,
    correlation_matrix,
    normalized_action,
    sample_action,
)
from src.training.sac.learner import initialize, update_checked
from tests.training.test_sac_learning import observation


def test_whitened_coordinates_match_independent_solve_and_roundtrip_at_all_g():
    rng = np.random.default_rng(6)
    speed = np.array([0.0, 0.1, 1.0, 2.0, 12.0])
    b = np.asarray(correlation_matrix())
    g = np.clip((speed / 2) ** 3, 0.002, 1)
    c = 0.02 * np.sqrt(2 * np.pi * np.e)
    expected = rng.normal(size=(5, 30, 2)) * 12
    raw = g[:, None, None] * c * np.einsum("ij,bjd->bid", b, expected)
    saved = raw.copy()
    actual = normalized_action(raw, speed)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    independent = np.stack([np.linalg.solve(b, a) for a in raw]) / g[:, None, None] / c
    np.testing.assert_allclose(actual, independent, rtol=1e-10, atol=1e-10)
    restored = g[:, None, None] * c * np.einsum("ij,bjd->bid", b, actual)
    np.testing.assert_allclose(restored, raw, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(raw, saved)
    assert actual.dtype == jnp.float64 and float(abs(actual).max()) > 1


def test_initial_gaussian_has_contract_whitened_variance_at_low_and_high_speed():
    speed = jnp.array([0.0, 0.1, 1.0, 10.0])
    epsilon = jax.random.normal(jax.random.PRNGKey(9), (4, 30, 2), dtype=jnp.float64)
    _, rho = Actor().apply(
        Actor().init(jax.random.PRNGKey(0), observation(4)), observation(4)
    )
    raw, _, _ = sample_action(
        jnp.zeros((4, 30, 2)), rho, speed, jax.random.PRNGKey(1), epsilon
    )
    np.testing.assert_allclose(
        normalized_action(raw, speed),
        epsilon / np.sqrt(2 * np.pi * np.e),
        rtol=1e-7,
        atol=1e-8,
    )


def test_normalization_gradient_matches_full_rank_inverse_adjoint():
    b = np.asarray(correlation_matrix())
    weights = np.linspace(-1, 1, 60).reshape(30, 2)
    g, c = 0.002, 0.02 * np.sqrt(2 * np.pi * np.e)
    gradient = jax.grad(lambda u: jnp.sum(normalized_action(u, 0.0) * weights))(
        jnp.zeros((30, 2), jnp.float64)
    )
    expected = np.linalg.solve(b.T, weights) / (g * c)
    np.testing.assert_allclose(gradient, expected, rtol=1e-10, atol=1e-5)
    assert np.linalg.matrix_rank(np.linalg.solve(b, np.eye(30))) == 30


def test_critic_coordinate_choice_keeps_parameter_layout_and_learns_through_actor():
    obs = observation(2)
    obs["features"]["ego"][:, 2] = np.array([0.5, 5.0]) / 30
    obs["field_masks"]["ego"][:, 2] = True
    raw = initialize(obs, critic=Critic(zero_output=True))
    state = initialize(
        obs, critic=Critic(zero_output=True, action_coordinates="whitened")
    )
    for a, b in zip(jax.tree_util.tree_leaves(raw), jax.tree_util.tree_leaves(state)):
        np.testing.assert_array_equal(a, b)
    batch = dict(
        observation=obs,
        next_observation=obs,
        raw_u=np.zeros((2, 30, 2), np.float64),
        reward=np.array([-1.0, -0.5]),
        delta_ticks=np.ones(2),
        bootstrap=np.zeros(2, bool),
    )
    updated, metrics = update_checked(state, batch)
    assert int(updated.updates) == 1 and int(updated.bc_updates) == 0
    assert float(metrics["actor_grad_norm"]) > 0 and bool(metrics["state_finite"])
    assert np.any(np.asarray(updated.actor.params["mu"]["kernel"]) != 0)


def test_unknown_critic_coordinates_are_rejected():
    with pytest.raises(ValueError, match="coordinates"):
        Critic(action_coordinates="wrong").init(
            jax.random.PRNGKey(0), observation(), jnp.zeros((2, 30, 2))
        )


def test_entropy_mean_coordinates_recover_head_output_across_speeds():
    obs = observation(3)
    speed = np.array([0.0, 1.0, 10.0])
    obs["features"]["ego"][:, 2] = speed / 30
    obs["field_masks"]["ego"][:, 2] = True
    actor = Actor(mean_basis="integrated", mean_coordinate_scale="entropy")
    variables = actor.init(jax.random.PRNGKey(1), obs)
    bias = jnp.linspace(-1, 1, 60, dtype=jnp.float32)
    variables["params"]["mu"]["bias"] = bias
    mu, rho = actor.apply(variables, obs)
    recovered = normalized_action(mu, speed)
    # f32 mean storage introduces cancellation when B is inverted in f64.
    np.testing.assert_allclose(
        recovered, np.tile(bias.reshape(1, 30, 2), (3, 1, 1)), rtol=3e-3, atol=3e-4
    )
    assert mu.dtype == jnp.float32
    plain = Actor(mean_basis="integrated")
    np.testing.assert_array_equal(rho, plain.apply(variables, obs)[1])
    variables["params"]["mu"]["bias"] = bias * 1e8
    large, _ = actor.apply(variables, obs)
    assert float(jnp.max(jnp.abs(large[0]))) > 1


def test_entropy_mean_gradient_preserves_all_coordinates_at_stationary_state():
    obs = observation(1)
    actor = Actor(mean_basis="integrated", mean_coordinate_scale="entropy")
    params = actor.init(jax.random.PRNGKey(1), obs)["params"]

    def forward(bias):
        changed = {**params, "mu": {**params["mu"], "bias": bias}}
        return normalized_action(
            actor.apply({"params": changed}, obs)[0], jnp.zeros(1)
        )[0].reshape(-1)

    jac = jax.jacfwd(forward)(jnp.zeros(60, jnp.float32))
    np.testing.assert_allclose(jac, np.eye(60), rtol=1e-3, atol=5e-4)
    assert np.linalg.matrix_rank(np.asarray(jac, np.float64)) == 60


def test_entropy_mean_requires_the_explicit_basis_amendment():
    with pytest.raises(ValueError, match="integrated"):
        Actor(mean_coordinate_scale="entropy").init(
            jax.random.PRNGKey(1), observation()
        )
