"""Mean basis precision must preserve NN state, sampling and differentiability."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from scripts.sac_diagnostic_actor import diagnostic_actor, actor_factory_hash
from src.training.sac.networks import Actor, correlation_matrix, sample_action
from tests.training.test_sac_learning import observation


def arm():
    return dict(mean_gain=1e-4, mean_basis="integrated", mean_coordinate_scale="raw",
                actor_head_layer_norm=True, mean_basis_precision="float64")


def test_legacy_factory_returns_original_actor_and_unknown_precision_fails():
    options = arm()
    options.pop("mean_basis_precision")
    assert type(diagnostic_actor(options, restoring=True)) is Actor
    for changed in ({"mean_basis_precision":"bad"}, {"mean_basis":"direct"}):
        with pytest.raises(ValueError, match="integrated"):
            diagnostic_actor(dict(arm(), **changed))
    with pytest.raises(ValueError, match="hash"):
        diagnostic_actor(arm(), restoring=True)
    diagnostic_actor(dict(arm(), actor_factory_sha256=actor_factory_hash()), restoring=True)


def test_initial_parameters_and_gaussian_are_identical():
    obs = observation(3)
    candidate = diagnostic_actor(arm())
    original = candidate.base
    key = jax.random.PRNGKey(3)
    old, new = original.init(key,obs), candidate.init(key,obs)
    assert serialization.to_bytes(old) == serialization.to_bytes(new)
    old_mu,old_rho = original.apply(old,obs)
    new_mu,new_rho = candidate.apply(new,obs)
    np.testing.assert_array_equal(old_mu,new_mu)
    np.testing.assert_array_equal(old_rho,new_rho)
    speed = jnp.array([0.,1.,15.])
    for old_value,new_value in zip(sample_action(old_mu,old_rho,speed,key),
                                   sample_action(new_mu,new_rho,speed,key)):
        np.testing.assert_array_equal(old_value,new_value)


def test_nonzero_head_roundtrip_and_gradient_match_independent_linear_map():
    obs = observation(2)
    actor = diagnostic_actor(arm())
    variables = actor.init(jax.random.PRNGKey(4),obs)
    params = copy.deepcopy(variables['params'])
    bias = jnp.asarray(np.sin(np.arange(60))*100, jnp.float32)
    params['mu']['bias'] = bias
    mu,rho = jax.jit(actor.apply)({'params':params},obs)
    b = np.asarray(correlation_matrix())
    scaled = np.asarray(bias*1e-4).reshape(30,2).astype(float)
    expected = b@scaled
    np.testing.assert_allclose(mu,np.broadcast_to(expected,(2,30,2)),atol=2e-15,rtol=1e-12)
    for row in np.asarray(mu):
        np.testing.assert_allclose(np.linalg.solve(b,row),scaled,atol=1e-12,rtol=1e-9)
    np.testing.assert_array_equal(rho,actor.base.apply({'params':params},obs)[1])
    assert mu.dtype == jnp.float64 and all(x.dtype==jnp.float32 for x in jax.tree_util.tree_leaves(params))
    weights = jnp.asarray(np.linspace(-1,1,60).reshape(30,2),jnp.float64)
    def loss(value):
        changed = dict(params,mu=dict(params['mu'],bias=value))
        return jnp.sum(actor.apply({'params':changed},obs)[0]*weights)
    actual = jax.grad(loss)(bias)
    expected_gradient = (2e-4*b.T@np.asarray(weights)).reshape(60)
    np.testing.assert_allclose(actual,expected_gradient,rtol=2e-6,atol=1e-11)


def test_full_learner_restore_and_real_update_preserve_source_state():
    from src.training.sac.learner import initialize, update_checked
    obs = observation(2)
    candidate = diagnostic_actor(arm())
    baseline = initialize(obs,seed=7,actor=candidate.base)
    initial = initialize(obs,seed=7,actor=candidate)
    original_bytes = serialization.to_bytes(baseline)
    assert serialization.to_bytes(initial) == original_bytes
    restored = serialization.from_bytes(initial,original_bytes)
    batch = dict(observation=obs,next_observation=obs,raw_u=np.zeros((2,30,2),np.float64),
                 reward=np.array([-20.,-.1]),bootstrap=np.zeros(2,bool),delta_ticks=np.ones(2,np.int32))
    changed,metrics = update_checked(restored,batch)
    assert int(changed.updates)==1 and int(changed.bc_updates)==0
    assert all(np.isfinite(float(v)) for v in metrics.values())
    assert serialization.to_bytes(baseline)==serialization.to_bytes(restored)==original_bytes
    assert any(not np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(changed.actor.params),
                                                     jax.tree_util.tree_leaves(restored.actor.params)))
