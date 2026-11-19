"""Check physical-coordinate invariance and differentiability without a rollout."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.sac_physical_critic import PhysicalActionCritic, physical_critic_input_copy
from src.training.sac.networks import Critic, correlation_matrix, normalized_action
from tests.training.test_sac_learning import observation


def test_internal_coordinates_remove_speed_factor_without_clipping_or_raw_mutation():
    speeds=jnp.asarray([0.,.5,2.,12.])
    raw=np.random.default_rng(7).normal(size=(4,30,2))
    before=raw.copy()
    actual=normalized_action(physical_critic_input_copy(raw,speeds),speeds)
    c=.02*np.sqrt(2*np.pi*np.e)
    basis=np.asarray(correlation_matrix())
    expected=np.stack([np.linalg.solve(basis,u)/c for u in raw])
    # B is ill-conditioned; compare normwise forward error and reconstruction,
    # not arbitrary elementwise relative error near cancellation coordinates.
    tolerance=10*np.finfo(np.float64).eps*np.linalg.cond(basis)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < tolerance
    reconstructed=np.einsum('ij,bjd->bid',basis,np.asarray(actual))*c
    assert np.linalg.norm(reconstructed-raw)/np.linalg.norm(raw) < tolerance
    np.testing.assert_array_equal(raw,before)
    # Full-rank linear map: a large action remains large and differentiable.
    doubled=normalized_action(physical_critic_input_copy(2*raw,speeds),speeds)
    np.testing.assert_allclose(doubled,2*actual,rtol=1e-12,atol=1e-9)


def test_same_parameter_layout_high_speed_behavior_and_nonzero_low_speed_gradient():
    obs=observation(2)
    obs['features']['ego'][:,2]=np.array([0.,10.])/30
    obs['field_masks']['ego'][:,2]=True
    raw=jnp.asarray(np.random.default_rng(1).normal(size=(2,30,2))*.0001)
    base=Critic(action_coordinates='whitened')
    physical=PhysicalActionCritic(action_coordinates='whitened')
    params=base.init(jax.random.PRNGKey(4),obs,raw)
    alternate=physical.init(jax.random.PRNGKey(4),obs,raw)
    for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(alternate)):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(physical.apply(params,obs,raw)[1],base.apply(params,obs,raw)[1])
    grad=jax.grad(lambda u: physical.apply(params,obs,u).sum())(raw)
    assert np.isfinite(grad).all() and np.any(np.asarray(grad[0])!=0)
    with pytest.raises(ValueError,match='whitened'):
        PhysicalActionCritic(action_coordinates='raw').init(jax.random.PRNGKey(4),obs,raw)
