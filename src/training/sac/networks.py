"""Mask-aware v1.2 encoder and unbounded correlated Gaussian waypoint actor.

The process entry point must enable JAX x64 before initializing these models.
Parameters/features are float32; action sampling and density are float64.
"""

import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from src.contracts.observation import CORE_SHAPES, ITEM_SHAPES


def require_x64():
    if not jax.config.x64_enabled:
        raise RuntimeError("waypoint SAC requires JAX_ENABLE_X64=true")


def masked_pool(values, valid):
    valid = valid[..., None]
    count = jnp.sum(valid, axis=-2)
    mean = jnp.sum(jnp.where(valid, values, 0.0), axis=-2) / jnp.maximum(count, 1)
    maximum = jnp.max(jnp.where(valid, values, -jnp.inf), axis=-2)
    return jnp.concatenate([mean, jnp.where(count > 0, maximum, 0.0)], axis=-1)


class ItemMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.relu(nn.Dense(64, dtype=jnp.float32, param_dtype=jnp.float32)(x))
        return x


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, observation):
        features, fields, items = (
            observation[k] for k in ("features", "field_masks", "item_masks")
        )
        outputs = []
        for name in CORE_SHAPES:
            x = features[name]
            x = jnp.where(fields[name], x, 0.0).astype(jnp.float32)
            x = jnp.concatenate([x, fields[name].astype(jnp.float32)], axis=-1)
            if name not in ITEM_SHAPES:
                outputs.append(x)
                continue
            mask = items[name]
            x = jnp.concatenate([x, mask[..., None].astype(jnp.float32)], axis=-1)
            if name == "agents":
                # Same object slot's three times, with every history/field mask.
                x = jnp.swapaxes(x, -3, -2)
                x = x.reshape((*x.shape[:-2], -1))
                mask = jnp.any(mask, axis=-2)
            encoded = ItemMLP(name=name + "_points")(x)
            pooled = masked_pool(encoded, mask)
            if name in ("lanes", "boundaries"):
                poly_valid = jnp.any(mask, axis=-1)
                poly = ItemMLP(name=name + "_polylines")(
                    jnp.concatenate(
                        [pooled, poly_valid[..., None].astype(jnp.float32)], axis=-1
                    )
                )
                pooled = masked_pool(poly, poly_valid)
            elif name == "route":
                # One route polyline. Preserve the same point -> polyline scheme.
                poly_valid = jnp.any(mask, axis=-1, keepdims=True)
                poly = ItemMLP(name="route_polyline")(
                    jnp.concatenate([pooled, poly_valid.astype(jnp.float32)], axis=-1)
                )
                pooled = masked_pool(poly[..., None, :], poly_valid)
            outputs.append(pooled)
        return jnp.concatenate(outputs, axis=-1)


class Actor(nn.Module):
    # Explicit diagnostic option: a full-rank linear gain, never action clipping
    # or temporal smoothing. Defaults retain the previous production actor.
    mean_gain: float = 1.0

    @nn.compact
    def __call__(self, observation):
        x = Encoder()(observation)
        for _ in range(2):
            x = nn.relu(nn.Dense(256, dtype=jnp.float32, param_dtype=jnp.float32)(x))
        mu = nn.Dense(
            60,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="mu",
            dtype=jnp.float32,
        )(x)
        rho = nn.Dense(
            2,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(math.log(1.5)),
            name="rho",
            dtype=jnp.float32,
        )(x)
        if self.mean_gain <= 0:
            raise ValueError("mean_gain must be positive")
        return (mu * self.mean_gain).reshape((*mu.shape[:-1], 30, 2)), rho


class Critic(nn.Module):
    # Explicit stability amendment: normalize hidden critic features, not actions.
    layer_norm: bool = True
    zero_output: bool = False

    @nn.compact
    def __call__(self, observation, raw_u):
        x = Encoder()(observation)
        # This cast retains the differentiable actor path; replay still stores f64.
        u = raw_u.astype(jnp.float32).reshape((*raw_u.shape[:-2], 60))
        x = jnp.concatenate([x, u], axis=-1)
        for _ in range(2):
            x = nn.Dense(256, dtype=jnp.float32, param_dtype=jnp.float32)(x)
            if self.layer_norm:
                x = nn.LayerNorm(dtype=jnp.float32, param_dtype=jnp.float32)(x)
            x = nn.relu(x)
        output_options = (
            {"kernel_init": nn.initializers.zeros_init()} if self.zero_output else {}
        )
        return nn.Dense(
            1, dtype=jnp.float32, param_dtype=jnp.float32, **output_options
        )(x)[..., 0]


def correlation_matrix():
    require_x64()
    i = jnp.arange(1, 31, dtype=jnp.float64)[:, None]
    j = jnp.arange(1, 31, dtype=jnp.float64)[None, :]
    return jnp.where(
        j <= i, 0.1**3 * (i - j + 1) * (i - j + 2) / (2 * 3 * (i * 0.1) ** 2), 0.0
    )


def sample_action(mu, rho, speed, key, epsilon=None):
    require_x64()
    mu = mu.astype(jnp.float64)
    sigma = 0.005 + 0.025 * jax.nn.sigmoid(rho.astype(jnp.float64))
    g = jnp.clip((jnp.asarray(speed, jnp.float64) / 2.0) ** 3, 0.002, 1.0)
    epsilon = (
        jax.random.normal(key, mu.shape, dtype=jnp.float64)
        if epsilon is None
        else jnp.asarray(epsilon, jnp.float64)
    )
    b = correlation_matrix()
    u = mu + g[..., None, None] * jnp.einsum(
        "ij,...jd->...id", b, sigma[..., None, :] * epsilon
    )
    ell = (
        30.0
        - 0.5 * jnp.sum(epsilon**2, axis=(-2, -1))
        - 30.0 * jnp.sum(jnp.log(sigma / 0.02), axis=-1)
    )
    log_pi = (
        -0.5 * jnp.sum(epsilon**2, axis=(-2, -1))
        - 30 * jnp.log(2 * jnp.pi)
        - 2 * jnp.sum(jnp.log(jnp.diag(b)))
        - 60 * jnp.log(g)
        - 30 * jnp.sum(jnp.log(sigma), axis=-1)
    )
    return u, ell, log_pi


def observation_speed(observation):
    # The environment normalizes the independently measured speed by 30 m/s.
    return observation["features"]["ego"][..., 2].astype(jnp.float64) * 30.0
