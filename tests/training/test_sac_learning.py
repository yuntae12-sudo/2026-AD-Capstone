import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from src.contracts.observation import CoreObservation
from src.training.sac.replay import (
    pack_observation,
    unpack_observation,
    Replay,
    Transition,
)
from src.training.sac.networks import (
    Actor,
    Encoder,
    correlation_matrix,
    sample_action,
    masked_pool,
)
from src.training.sac.learner import (
    initialize,
    bc_update,
    targets,
    update_checked,
    update,
)
from src.training.sac.checkpoint import save_checkpoint, load_checkpoint


def observation(batch=2):
    core = CoreObservation.empty()
    packed = pack_observation(core)
    return unpack_observation(*(np.repeat(a[None], batch, axis=0) for a in packed))


def test_mask_pool_and_encoder_order_are_stable_under_jit():
    x = jnp.array([[[2.0, -2.0], [100.0, 100.0]], [[3.0, 4.0], [5.0, 6.0]]])
    mask = jnp.array([[True, False], [False, False]])
    np.testing.assert_array_equal(
        masked_pool(x, mask), [[2.0, -2.0, 2.0, -2.0], [0.0, 0.0, 0.0, 0.0]]
    )
    obs = observation()
    # Valid scalar feature makes an accidental dict-order change observable.
    obs["features"]["ego"][:, 2] = 0.2
    obs["field_masks"]["ego"][:, 2] = True
    encoder = Encoder()
    params = encoder.init(jax.random.PRNGKey(1), obs)
    a = encoder.apply(params, obs)
    b = jax.jit(encoder.apply)(params, obs)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert np.isfinite(a).all()
    # Invalid fields cannot affect the embedding; all masks still reach the NN.
    obs["features"]["agents"].fill(100.0)
    np.testing.assert_allclose(encoder.apply(params, obs), a, atol=1e-6)


def test_correlated_density_independent_solve_and_gradient():
    rng = np.random.default_rng(5)
    mu = jnp.zeros((30, 2), jnp.float32)
    rho = jnp.array([np.log(1.5), np.log(1.5)], jnp.float32)
    epsilon = jnp.asarray(rng.normal(size=(30, 2)))
    u, ell, logpi = sample_action(
        mu, rho, jnp.array(0.5), jax.random.PRNGKey(2), epsilon
    )
    b = np.asarray(correlation_matrix())
    g = (0.5 / 2) ** 3
    sigma = 0.005 + 0.025 / (1 + np.exp(-np.asarray(rho, np.float64)))
    recovered = np.linalg.solve(g * b, np.asarray(u) - np.asarray(mu)) / sigma
    independent = (
        -0.5 * np.sum(recovered**2)
        - 30 * np.log(2 * np.pi)
        - 2 * np.linalg.slogdet(b)[1]
        - 60 * np.log(g)
        - 30 * np.log(sigma).sum()
    )
    assert float(logpi) == pytest.approx(independent, abs=1e-8)
    c = 0.02 * np.sqrt(2 * np.pi * np.e)
    assert float(ell) == pytest.approx(
        float(logpi) + 2 * np.linalg.slogdet(b)[1] + 60 * np.log(g) + 60 * np.log(c),
        abs=1e-10,
    )
    assert u.dtype == jnp.float64
    shifted = sample_action(
        mu + 2.0, rho, jnp.array(0.5), jax.random.PRNGKey(2), epsilon
    )[0]
    np.testing.assert_allclose(shifted - u, 2.0, atol=1e-15)
    gradient = jax.grad(
        lambda m: jnp.sum(
            sample_action(m, rho, jnp.array(5.0), jax.random.PRNGKey(2), epsilon)[
                0
            ].astype(jnp.float32)
        )
    )(mu)
    np.testing.assert_array_equal(gradient, np.ones((30, 2), np.float32))


def test_bc_sac_update_terminal_target_and_checkpoint(tmp_path, monkeypatch):
    obs = observation()
    state = initialize(obs, seed=2)
    assert all(
        x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(state.actor.params)
    )
    mu, rho = state.actor.apply_fn({"params": state.actor.params}, obs)
    np.testing.assert_array_equal(mu, 0.0)
    np.testing.assert_allclose(rho, np.log(1.5), rtol=1e-6)
    batch = dict(
        observation=obs,
        next_observation=obs,
        raw_u=np.full((2, 30, 2), 0.01),
        reward=np.array([0.1, -20.0]),
        delta_ticks=np.array([1, 2]),
        bootstrap=np.array([True, False]),
        teacher_accepted=np.ones(2, bool),
    )
    bc, loss = bc_update(state, batch)
    assert float(loss) > 0 and int(bc.bc_updates) == 1
    for a, b in zip(
        jax.tree_util.tree_leaves(state.actor.params["rho"]),
        jax.tree_util.tree_leaves(bc.actor.params["rho"]),
    ):
        np.testing.assert_array_equal(a, b)
    terminal = dict(batch, bootstrap=np.zeros(2, bool))
    import src.training.sac.learner as learner

    with monkeypatch.context() as patch:
        patch.setattr(
            learner,
            "_bootstrapped_value",
            lambda *a: (_ for _ in ()).throw(
                AssertionError("terminal target actor called")
            ),
        )
        np.testing.assert_array_equal(
            targets(bc, terminal, jax.random.PRNGKey(0)), batch["reward"]
        )
    updated, metrics = update_checked(bc, batch)
    assert int(updated.updates) == 1 and all(
        np.isfinite(float(v)) for v in metrics.values()
    )
    for old, new, online in zip(
        jax.tree_util.tree_leaves(bc.target1),
        jax.tree_util.tree_leaves(updated.target1),
        jax.tree_util.tree_leaves(updated.q1.params),
    ):
        np.testing.assert_allclose(new, 0.995 * old + 0.005 * online, atol=1e-7)
    with pytest.raises(ValueError):
        update(bc, batch, replay_size=4999)
    versions = {"schema": 3, "distribution": "v2"}
    replay = Replay(2, versions=versions)
    core = CoreObservation.empty()
    replay.add(
        Transition(
            core,
            np.zeros((30, 2)),
            0.0,
            core,
            1,
            False,
            True,
            True,
            "episode",
            "scene",
            0,
            True,
        )
    )
    path = tmp_path / "checkpoint"
    with pytest.raises(ValueError):
        save_checkpoint(path, updated, replay, versions, {}, episode_boundary=False)
    save_checkpoint(
        path, updated, replay, versions, {"next_scene": 2}, episode_boundary=True
    )
    restored, loaded, run = load_checkpoint(path, state, versions)
    assert int(restored.updates) == 1 and loaded.size == 1 and run["next_scene"] == 2
    # RNG, optimizer moments, targets and params resume the identical next update.
    a, _ = update_checked(updated, batch)
    b, _ = update_checked(restored, batch)
    for av, bv in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(av, bv)
