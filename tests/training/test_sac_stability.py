import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.training.sac.learner import (
    gradient_norm,
    clip_gradients,
    initialize,
    update_checked,
)
from src.training.sac.networks import Critic
from tests.training.test_sac_learning import observation


@pytest.mark.parametrize("magnitude", [1e20, 1e38])
def test_large_finite_float32_gradients_clip_without_norm_overflow(magnitude):
    gradients = {
        "a": jnp.full((1024,), magnitude, jnp.float32),
        "b": jnp.array([-magnitude], jnp.float32),
    }
    norm = gradient_norm(gradients)
    assert np.isfinite(float(norm))
    tx = clip_gradients()
    clipped, _ = tx.update(gradients, tx.init(gradients))
    assert float(gradient_norm(clipped)) == pytest.approx(10.0, rel=1e-6)
    assert all(x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(clipped))
    assert float(clipped["a"][0]) > 0 and float(clipped["b"][0]) < 0


def test_clipping_preserves_small_gradients_and_zero():
    for values in (jnp.array([3.0, 4.0], jnp.float32), jnp.zeros(3, jnp.float32)):
        tx = clip_gradients()
        out, _ = tx.update({"x": values}, tx.init({"x": values}))
        np.testing.assert_array_equal(out["x"], values)


def test_critic_normalization_stops_linear_action_scale_extrapolation():
    obs = observation()
    u = jnp.ones((2, 30, 2), jnp.float32)
    normalized = Critic()
    params = normalized.init(jax.random.PRNGKey(2), obs, u)
    low = normalized.apply(params, obs, u * 1e6)
    high = normalized.apply(params, obs, u * 1e8)
    np.testing.assert_allclose(low, high, atol=1e-4, rtol=1e-4)
    assert np.max(np.abs(high)) < 100
    raw = Critic(layer_norm=False)
    params = raw.init(jax.random.PRNGKey(2), obs, u)
    low = raw.apply(params, obs, u * 1e6)
    high = raw.apply(params, obs, u * 1e8)
    assert np.max(np.abs(high)) > np.max(np.abs(low)) * 50


def test_nonfinite_target_parameters_are_rejected_even_with_terminal_batch():
    obs = observation()
    s = initialize(obs)
    s = s.replace(
        target1=jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.inf), s.target1)
    )
    batch = dict(
        observation=obs,
        next_observation=obs,
        raw_u=np.zeros((2, 30, 2)),
        reward=np.zeros(2),
        delta_ticks=np.ones(2),
        bootstrap=np.zeros(2, bool),
    )
    with pytest.raises(FloatingPointError, match="state_finite=False"):
        update_checked(s, batch)
