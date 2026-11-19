"""Read-only decomposition using the production fused actor/target-Q path."""

import jax
import jax.numpy as jnp
from src.training.sac.learner import draw


@jax.jit
def target_components(state, next_observation, key):
    u, ell = draw(state.actor, next_observation, key)
    q1 = state.q1.apply_fn({"params": state.target1}, next_observation, u)
    q2 = state.q2.apply_fn({"params": state.target2}, next_observation, u)
    q = jnp.minimum(q1, q2)
    correction = -jnp.exp(state.log_alpha) * ell
    return q, correction, q - jnp.exp(state.log_alpha) * ell
