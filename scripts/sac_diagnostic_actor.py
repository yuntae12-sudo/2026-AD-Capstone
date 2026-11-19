"""Opt-in numerical comparison for the approved full-rank B mean.

The production Actor, parameter layout and Gaussian sampler are unchanged.
Only the B contraction uses float64; the NN and gain multiplication stay f32.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp

from src.training.sac.networks import (
    Actor, correlation_matrix, observation_speed, require_x64,
)


def actor_factory_hash():
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).resolve().parents[1] / "src/training/sac/networks.py"):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Float64MeanBasisActor:
    """Reuse the exact NN with a captured pre-basis mean, without new parameters."""

    base: Actor

    def init(self, key, observation):
        return self.base.init(key, observation)

    def apply(self, variables, observation):
        require_x64()
        (_, rho), captured = self.base.apply(
            variables, observation,
            capture_intermediates=lambda module, method: module.scope.path == ("mu",),
            mutable=["intermediates"],
        )
        head = captured["intermediates"]["mu"]["__call__"][0]
        # Preserve the original f32 gain operation before changing only B's dot.
        scaled = (head * self.base.mean_gain).reshape((*head.shape[:-1], 30, 2))
        mu = jnp.einsum(
            "ij,...jd->...id", correlation_matrix(), scaled.astype(jnp.float64),
            precision=jax.lax.Precision.HIGHEST,
        )
        if self.base.mean_coordinate_scale == "entropy":
            g = jnp.clip((observation_speed(observation) / 2.0) ** 3, 0.002, 1.0)
            scale = (g * (0.02 * jnp.sqrt(2 * jnp.pi * jnp.e))).astype(jnp.float32)
            mu = mu * scale[..., None, None]
        return mu, rho


def diagnostic_actor(arm, *, restoring=False):
    base = Actor(
        mean_gain=arm.get("mean_gain", 1.0),
        mean_basis=arm.get("mean_basis", "direct"),
        head_layer_norm=arm.get("actor_head_layer_norm", False),
        mean_coordinate_scale=arm.get("mean_coordinate_scale", "raw"),
    )
    precision = arm.get("mean_basis_precision", "default")
    if precision == "default":
        return base
    if precision != "float64" or base.mean_basis != "integrated":
        raise ValueError("float64 mean basis requires the integrated B mean")
    if restoring and arm.get("actor_factory_sha256") != actor_factory_hash():
        raise ValueError("actor mean-basis implementation hash mismatch")
    return Float64MeanBasisActor(base)
