import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.sac_critic_initialization import zero_critic_action_initialization
from src.training.sac.learner import initialize, update_checked
from src.training.sac.networks import Actor, Critic
from tests.training.test_sac_learning import observation


def test_zero_action_initialization_keeps_targets_and_learns_action_dependence():
    obs = observation()
    # Real SAC observations have valid ego fields; the empty schema fixture is
    # entirely masked and would leave every ReLU at zero after this initializer.
    obs["features"]["ego"][:, 2] = [0.1, 0.2]
    obs["field_masks"]["ego"][:, 2] = True
    original = initialize(
        obs, actor=Actor(mean_gain=1e-4), critic=Critic(zero_output=True)
    )
    state = zero_critic_action_initialization(original)
    for old, q, target in [
        (original.q1, state.q1, state.target1),
        (original.q2, state.q2, state.target2),
    ]:
        old_kernel, new_kernel = (
            old.params["Dense_0"]["kernel"],
            q.params["Dense_0"]["kernel"],
        )
        assert np.any(np.asarray(old_kernel[-60:]) != 0)
        np.testing.assert_array_equal(new_kernel[-60:], 0)
        np.testing.assert_array_equal(new_kernel[:-60], old_kernel[:-60])
        for a, b in zip(
            jax.tree_util.tree_leaves(q.params), jax.tree_util.tree_leaves(target)
        ):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(old.opt_state),
            jax.tree_util.tree_leaves(q.opt_state),
        ):
            np.testing.assert_array_equal(a, b)
    for a, b in zip(
        jax.tree_util.tree_leaves(original.actor),
        jax.tree_util.tree_leaves(state.actor),
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(original.rng, state.rng)
    raw = jnp.full((2, 30, 2), 0.01, dtype=jnp.float64)
    batch = dict(
        observation=obs,
        next_observation=obs,
        raw_u=raw,
        reward=np.array([-1.0, -2.0]),
        delta_ticks=np.ones(2),
        bootstrap=np.zeros(2, bool),
    )
    for _ in range(3):
        state, metrics = update_checked(state, batch)
        assert bool(metrics["state_finite"])
    assert np.any(np.asarray(state.q1.params["Dense_0"]["kernel"][-60:]) != 0)
    derivative = jax.grad(
        lambda u: jnp.sum(state.q1.apply_fn({"params": state.q1.params}, obs, u))
    )(raw)
    assert np.isfinite(derivative).all() and np.any(np.asarray(derivative) != 0)
    with pytest.raises(ValueError, match="fresh"):
        zero_critic_action_initialization(state)
