import jax
import numpy as np
from scripts.sac_target_components import target_components
from src.training.sac.learner import initialize, _bootstrapped_value
from tests.training.test_sac_learning import observation


def test_fused_components_reconstruct_production_soft_value():
    obs = observation(2)
    obs['features']['ego'][:,2] = [0., .3]
    obs['field_masks']['ego'][:,2] = True
    state = initialize(obs)
    key = jax.random.PRNGKey(7)
    q, correction, total = target_components(state, obs, key)
    expected = _bootstrapped_value(state, obs, key)
    np.testing.assert_allclose(total, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(total, q+correction, rtol=1e-10, atol=1e-10)
