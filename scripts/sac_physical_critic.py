"""Opt-in critic feature experiment; raw policy/replay/entropy stay unchanged.

The existing whitened critic divides its input copy by g(state). Multiplying
that copy by the same positive g first gives B^-1 u / c internally. This is a
state-conditioned, invertible feature map, not repair of the executed action.
"""

import jax.numpy as jnp

from src.training.sac.networks import Critic, observation_speed, require_x64


def physical_critic_input_copy(raw_u, speed):
    require_x64()
    raw = jnp.asarray(raw_u, jnp.float64)
    speed = jnp.asarray(speed, jnp.float64)
    g = jnp.clip((speed / 2.0) ** 3, 0.002, 1.0)
    return raw * g[..., None, None]


class PhysicalActionCritic(Critic):
    def __call__(self, observation, raw_u):
        if self.action_coordinates != "whitened":
            raise ValueError("physical feature experiment requires whitened base")
        copied = physical_critic_input_copy(raw_u, observation_speed(observation))
        return super().__call__(observation, copied)
