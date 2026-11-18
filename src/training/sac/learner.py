"""Independent actor/twin-Q SAC with section-10 update order and BC warmup."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from .networks import Actor, Critic, observation_speed, require_x64, sample_action


@struct.dataclass
class LearnerState:
    actor: TrainState
    q1: TrainState
    q2: TrainState
    target1: object
    target2: object
    log_alpha: object
    alpha_optimizer: object
    rng: object
    updates: int = 0
    bc_updates: int = 0


def gradient_norm(tree):
    """Accumulate squared float32 gradients in float64 before clipping."""
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.square(x.astype(jnp.float64)))
            for x in jax.tree_util.tree_leaves(tree)
        )
    )


def clip_gradients(max_norm=10.0):
    def transform(updates, state, params=None):
        del params
        norm = gradient_norm(updates)
        scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-300))
        return (
            jax.tree_util.tree_map(
                lambda x: (x.astype(jnp.float64) * scale).astype(x.dtype), updates
            ),
            state,
        )

    return optax.GradientTransformation(optax.init_empty_state, transform)


def optimizer():
    return optax.chain(clip_gradients(10.0), optax.adam(3e-4))


def initialize(observation, seed=0, *, actor=None, critic=None):
    require_x64()
    actor = Actor() if actor is None else actor
    critic = Critic() if critic is None else critic
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    mu = actor.init(keys[0], observation)["params"]
    action = jnp.zeros((*observation["features"]["ego"].shape[:-1], 30, 2), jnp.float64)
    q1 = critic.init(keys[1], observation, action)["params"]
    q2 = critic.init(keys[2], observation, action)["params"]
    alpha = jnp.asarray(np.log(0.001), jnp.float32)
    return LearnerState(
        TrainState.create(apply_fn=actor.apply, params=mu, tx=optimizer()),
        TrainState.create(apply_fn=critic.apply, params=q1, tx=optimizer()),
        TrainState.create(apply_fn=critic.apply, params=q2, tx=optimizer()),
        q1,
        q2,
        alpha,
        optimizer().init(alpha),
        keys[3],
    )


def draw(actor, observation, key):
    mu, rho = actor.apply_fn({"params": actor.params}, observation)
    u, ell, _ = sample_action(mu, rho, observation_speed(observation), key)
    return u, ell


@jax.jit
def _bootstrapped_value(state, next_observation, key):
    u, ell = draw(state.actor, next_observation, key)
    q1 = state.q1.apply_fn({"params": state.target1}, next_observation, u)
    q2 = state.q2.apply_fn({"params": state.target2}, next_observation, u)
    return jnp.minimum(q1, q2) - jnp.exp(state.log_alpha) * ell


def targets(state, batch, key):
    """Compact on host before any next-actor call; terminal rows are never drawn."""
    indices = np.flatnonzero(np.asarray(batch["bootstrap"]))
    result = jnp.asarray(batch["reward"], jnp.float64)
    if len(indices):
        next_obs = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x)[indices], batch["next_observation"]
        )
        value = _bootstrapped_value(state, next_obs, key)
        result = result.at[indices].add(
            0.99 ** jnp.asarray(batch["delta_ticks"])[indices] * value
        )
    return jax.lax.stop_gradient(result)


@jax.jit
def _update(state, observation, raw_u, target, actor_key, next_rng):
    def q_loss(params, apply):
        estimate = apply({"params": params}, observation, raw_u)
        return jnp.mean((estimate - target) ** 2)

    q1_loss, g1 = jax.value_and_grad(lambda p: q_loss(p, state.q1.apply_fn))(
        state.q1.params
    )
    q2_loss, g2 = jax.value_and_grad(lambda p: q_loss(p, state.q2.apply_fn))(
        state.q2.params
    )
    q1, q2 = state.q1.apply_gradients(grads=g1), state.q2.apply_gradients(grads=g2)

    def actor_loss(params):
        actor = state.actor.replace(params=params)
        u, ell = draw(actor, observation, actor_key)
        # Differentiate only actor params, while preserving dQ/du.
        qa = q1.apply_fn({"params": q1.params}, observation, u)
        qb = q2.apply_fn({"params": q2.params}, observation, u)
        return jnp.mean(jnp.exp(state.log_alpha) * ell - jnp.minimum(qa, qb)), (
            ell,
            jnp.max(jnp.abs(u)),
            jnp.max(jnp.abs(qa)),
            jnp.max(jnp.abs(qb)),
        )

    (actor_loss_value, (ell, action_max, q1_max, q2_max)), ga = jax.value_and_grad(
        actor_loss, has_aux=True
    )(state.actor.params)
    actor = state.actor.apply_gradients(grads=ga)
    alpha_grad = -jnp.mean(jax.lax.stop_gradient(ell)).astype(jnp.float32)
    alpha_updates, alpha_optimizer = optimizer().update(
        alpha_grad, state.alpha_optimizer, state.log_alpha
    )
    unbounded = optax.apply_updates(state.log_alpha, alpha_updates)
    log_alpha = jnp.clip(unbounded, jnp.log(1e-6), jnp.log(0.01)).astype(jnp.float32)
    polyak = lambda target, online: 0.995 * target + 0.005 * online
    updated = state.replace(
        actor=actor,
        q1=q1,
        q2=q2,
        target1=jax.tree_util.tree_map(polyak, state.target1, q1.params),
        target2=jax.tree_util.tree_map(polyak, state.target2, q2.params),
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        rng=next_rng,
        updates=state.updates + 1,
    )
    metrics = dict(
        q1_loss=q1_loss,
        q2_loss=q2_loss,
        actor_loss=actor_loss_value,
        ell=jnp.mean(ell),
        alpha=jnp.exp(log_alpha),
        alpha_at_bound=(unbounded <= jnp.log(1e-6)) | (unbounded >= jnp.log(0.01)),
        actor_grad_norm=gradient_norm(ga),
        q1_grad_norm=gradient_norm(g1),
        q2_grad_norm=gradient_norm(g2),
        actor_action_abs_max=action_max,
        actor_q1_abs_max=q1_max,
        actor_q2_abs_max=q2_max,
        target_abs_max=jnp.max(jnp.abs(target)),
        state_finite=jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(updated)]
            )
        ),
    )
    return updated, metrics


def update(state, batch, replay_size):
    if replay_size < 5000:
        raise ValueError("SAC requires at least 5000 valid replay transitions")
    if len(batch["reward"]) != 256:
        raise ValueError("contract SAC minibatch size is 256")
    return update_checked(state, batch)


def update_checked(state, batch):
    """Math/integration test entry point; production must call update()."""
    for name in ("observation", "next_observation", "raw_u", "reward"):
        if not all(
            np.isfinite(np.asarray(x)).all()
            for x in jax.tree_util.tree_leaves(batch[name])
        ):
            raise ValueError("corrupt learner input; update stopped")
    rng, target_key, actor_key = jax.random.split(state.rng, 3)
    target = targets(state, batch, target_key)
    updated, metrics = _update(
        state, batch["observation"], batch["raw_u"], target, actor_key, rng
    )
    if not bool(metrics["state_finite"]) or not all(
        np.isfinite(np.asarray(x)).all() for x in metrics.values()
    ):
        invalid = [
            name
            for name, value in metrics.items()
            if not np.isfinite(np.asarray(value)).all()
        ]
        raise FloatingPointError(
            f"nonfinite SAC update; old state retained; invalid_metrics={invalid}; state_finite={bool(metrics['state_finite'])}"
        )
    return updated, metrics


@jax.jit
def _bc_update(state, observation, raw_u):
    def loss(params):
        mu, _ = state.actor.apply_fn({"params": params}, observation)
        return jnp.mean((mu.astype(jnp.float64) - raw_u) ** 2)

    value, grad = jax.value_and_grad(loss)(state.actor.params)
    return state.replace(
        actor=state.actor.apply_gradients(grads=grad), bc_updates=state.bc_updates + 1
    ), (value, gradient_norm(grad))


def bc_update(state, batch):
    if int(state.updates):
        raise ValueError("BC may only precede SAC")
    if not np.asarray(batch["teacher_accepted"]).all():
        raise ValueError("BC requires accepted teacher paths")
    for name in ("observation", "raw_u"):
        if not all(
            np.isfinite(np.asarray(x)).all()
            for x in jax.tree_util.tree_leaves(batch[name])
        ):
            raise ValueError("corrupt BC input")
    updated, (loss, norm) = _bc_update(state, batch["observation"], batch["raw_u"])
    if not np.isfinite(float(loss)) or not np.isfinite(float(norm)):
        raise FloatingPointError("nonfinite BC loss/gradient")
    return updated, loss
