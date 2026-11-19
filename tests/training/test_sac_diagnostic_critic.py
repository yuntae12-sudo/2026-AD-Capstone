import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.sac_diagnostic_critic import diagnostic_critic, InputGainCritic, critic_factory_hash
from src.training.sac.networks import Critic
from tests.training.test_sac_learning import observation


def test_input_gain_retains_layout_and_exact_action_chain_rule():
    obs = observation(2)
    obs['features']['ego'][:,2] = np.array([0.,10.])/30
    obs['field_masks']['ego'][:,2] = True
    raw = jnp.asarray(np.random.default_rng(1).normal(size=(2,30,2))*.0001)
    original = np.asarray(raw).copy()
    base = Critic(action_coordinates='whitened')
    scaled = InputGainCritic(action_coordinates='whitened',action_gain=.1)
    variables=base.init(jax.random.PRNGKey(4),obs,raw)
    other=scaled.init(jax.random.PRNGKey(4),obs,raw)
    for a,b in zip(jax.tree_util.tree_leaves(variables),jax.tree_util.tree_leaves(other)):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(scaled.apply(variables,obs,raw),base.apply(variables,obs,raw*.1),rtol=0,atol=0)
    grad=jax.grad(lambda u: scaled.apply(variables,obs,u).sum())(raw)
    ref=jax.grad(lambda u: base.apply(variables,obs,u).sum())(raw*.1)*.1
    np.testing.assert_allclose(grad,ref,rtol=1e-6,atol=1e-6)
    assert np.isfinite(grad).all() and np.any(np.asarray(grad)!=0)
    np.testing.assert_array_equal(raw,original)


def test_old_artifact_uses_original_critic_and_invalid_gains_fail():
    assert type(diagnostic_critic(dict(zero_q=True))) is Critic
    for gain in [0.,-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError,match='positive'):
            diagnostic_critic(dict(zero_q=True,critic_action_gain=gain))


def test_scaled_restore_requires_recorded_implementation():
    arm=dict(zero_q=True,critic_action_gain=.1)
    with pytest.raises(ValueError,match='hash mismatch'):
        diagnostic_critic(arm,restoring=True)
    arm['critic_factory_sha256']=critic_factory_hash()
    assert isinstance(diagnostic_critic(arm,restoring=True),InputGainCritic)


def test_archived_gain_restore_retains_legacy_module_but_cannot_restore_physical():
    legacy = "9986af114dd291e6664e007782287c07d0e75bcfee9c2d05526b08b384a8ed12"
    arm = dict(zero_q=True, critic_action_gain=.1, critic_factory_sha256=legacy)
    assert isinstance(diagnostic_critic(arm, restoring=True), InputGainCritic)
    arm.update(critic_action_gain=1., critic_action_coordinates="physical_whitened")
    with pytest.raises(ValueError, match="hash mismatch"):
        diagnostic_critic(arm, restoring=True)


def test_physical_factory_requires_current_provenance_and_single_coordinate_change():
    from scripts.sac_physical_critic import PhysicalActionCritic
    arm = dict(zero_q=True, critic_action_coordinates="physical_whitened",
               critic_factory_sha256=critic_factory_hash())
    assert type(diagnostic_critic(arm, restoring=True)) is PhysicalActionCritic
    arm["critic_action_gain"] = .1
    with pytest.raises(ValueError, match="cannot combine"):
        diagnostic_critic(arm)
