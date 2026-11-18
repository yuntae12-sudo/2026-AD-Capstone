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
    # Opt-in research amendment approved on 2026-11-18, not production defaults.
    mean_basis: str = "direct"
    head_layer_norm: bool = False
    mean_coordinate_scale: str = "raw"

    @nn.compact
    def __call__(self, observation):
        x = Encoder()(observation)
        for _ in range(2):
            x = nn.Dense(256, dtype=jnp.float32, param_dtype=jnp.float32)(x)
            if self.head_layer_norm:
                x = nn.LayerNorm(dtype=jnp.float32, param_dtype=jnp.float32)(x)
            x = nn.relu(x)
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
        if not math.isfinite(self.mean_gain) or self.mean_gain <= 0:
            raise ValueError("mean_gain must be positive")
        mu = (mu * self.mean_gain).reshape((*mu.shape[:-1], 30, 2))
        if self.mean_basis == "integrated":
            # Reparameterize the mean before the unchanged Gaussian sampler.
            # B is full rank, preserving all 60 unbounded degrees of freedom.
            mu = jnp.einsum(
                "ij,...jd->...id", correlation_matrix().astype(jnp.float32), mu
            )
        elif self.mean_basis != "direct":
            raise ValueError("unknown mean basis")
        if self.mean_coordinate_scale == "entropy":
            if self.mean_basis != "integrated":
                raise ValueError("entropy mean coordinates require integrated basis")
            g = jnp.clip((observation_speed(observation) / 2.0) ** 3, 0.002, 1.0)
            scale = (g * (0.02 * math.sqrt(2 * math.pi * math.e))).astype(jnp.float32)
            mu = mu * scale[..., None, None]
        elif self.mean_coordinate_scale != "raw":
            raise ValueError("unknown mean coordinate scale")
        return mu, rho


class Critic(nn.Module):
    # Explicit stability amendment: normalize hidden critic features, not actions.
    layer_norm: bool = True
    zero_output: bool = False
    # Section 5 entropy coordinates, opt-in; replay/execution remain raw u.
    action_coordinates: str = "raw"

    @nn.compact
    def __call__(self, observation, raw_u):
        x = Encoder()(observation)
        # This cast retains the differentiable actor path; replay still stores f64.
        if self.action_coordinates == "raw":
            action = raw_u
        elif self.action_coordinates == "whitened":
            action = normalized_action(raw_u, observation_speed(observation))
        else:
            raise ValueError("unknown critic action coordinates")
        u = action.astype(jnp.float32).reshape((*raw_u.shape[:-2], 60))
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


def normalized_action(raw_u, speed):
    """Invert section-5 u = g B c w in f64 before critic's f32 input cast.

    No action repair, clipping, covariance inverse, or stop_gradient. c is the
    contract's entropy-coordinate scale, not the learned sigma.
    """
    require_x64()
    from jax.scipy.linalg import solve_triangular

    u = jnp.asarray(raw_u, jnp.float64)
    if u.shape[-2:] != (30, 2):
        raise ValueError("raw action must end in [30,2]")
    b = correlation_matrix()
    solved = jax.vmap(lambda a: solve_triangular(b, a, lower=True))(
        u.reshape((-1, 30, 2))
    ).reshape(u.shape)
    g = jnp.clip((jnp.asarray(speed, jnp.float64) / 2.0) ** 3, 0.002, 1.0)
    c = 0.02 * jnp.sqrt(2 * jnp.pi * jnp.e)
    return solved / (g[..., None, None] * c)


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
