"""The diagnostic options must not bound actions, smooth means, or freeze Q."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.training.sac.networks import Actor, Critic
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
