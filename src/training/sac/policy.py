"""Policy boundary preserving the sampled raw float64 action exactly."""

import jax
import jax.numpy as jnp
import numpy as np
from src.contracts.planning import WaypointProposal, decode_waypoints, future_times
from .networks import sample_action, observation_speed
from .replay import pack_observation, unpack_observation


def tensor_observation(core):
    return unpack_observation(*(jnp.asarray(a)[None] for a in pack_observation(core)))


def proposal_from_action(e, token, raw_u, version="sac-v1.2", sequence=0):
    ego = e.planning_anchor(token)
    u = np.asarray(raw_u, np.float64)
    xy = decode_waypoints(u, ego.position, ego.yaw, ego.velocity, ego.acceleration)
    return WaypointProposal(
        token,
        e.command.command_id,
        f"{token.episode_uid}:{token.sim_tick}:{sequence}",
        token.sim_tick,
        ego.position,
        ego.yaw,
        ego.velocity,
        ego.acceleration,
        ego.acceleration_valid,
        future_times(token.sim_tick),
        xy,
        u,
        model_version=version,
    )


class Policy:
    def __init__(self, actor, seed=0, deterministic=False):
        self.actor = actor
        self.rng = jax.random.PRNGKey(seed)
        self.deterministic = deterministic
        self.forward = jax.jit(actor.apply_fn)

    def action(self, core):
        observation = tensor_observation(core)
        mu, rho = self.forward({"params": self.actor.params}, observation)
        if self.deterministic:
            return np.asarray(mu[0], np.float64)
        self.rng, key = jax.random.split(self.rng)
        u, _, _ = sample_action(mu, rho, observation_speed(observation), key)
        return np.asarray(u[0], np.float64)

    def propose(self, e, token):
        e._require(token)
        return proposal_from_action(
            e, token, self.action(e._boundary().sac_observation)
        )
