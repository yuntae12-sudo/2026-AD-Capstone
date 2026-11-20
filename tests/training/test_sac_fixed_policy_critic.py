import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
from flax import serialization

from scripts.sac_fixed_policy_critic import critic_evaluation_step
from src.contracts.observation import CoreObservation
from src.training.sac.learner import initialize, update
from src.training.sac.replay import pack_observation, unpack_observation


@pytest.fixture
def example():
    packed = pack_observation(CoreObservation.empty())
    obs = unpack_observation(*(np.repeat(x[None], 256, axis=0) for x in packed))
    state = initialize(obs, seed=57)
    batch = dict(observation=obs, next_observation=obs,
                 raw_u=np.full((256, 30, 2), .001, dtype=np.float64),
                 reward=np.tile([.01, -20.0], 128),
                 delta_ticks=np.ones(256, dtype=np.int32),
                 bootstrap=np.tile([True, False], 128),
                 teacher_accepted=np.zeros(256, dtype=bool))
    return state, batch


def test_critic_matches_canonical_while_actor_temperature_stay_frozen(example):
    state, batch = example
    original = serialization.to_bytes(state)
    frozen_actor = serialization.to_bytes(state.actor)
    frozen_temperature = serialization.to_bytes((state.log_alpha, state.alpha_optimizer))
    for step in range(2):
        canonical, expected_metrics = update(state, batch, replay_size=6000 + step)
        actual, metrics = critic_evaluation_step(state, batch, replay_size=6000 + step)
        for key in ("q1", "q2", "target1", "target2", "rng"):
            assert serialization.to_bytes(getattr(actual, key)) == serialization.to_bytes(getattr(canonical, key))
        assert serialization.to_bytes(actual.actor) == frozen_actor
        assert serialization.to_bytes((actual.log_alpha, actual.alpha_optimizer)) == frozen_temperature
        assert serialization.to_bytes(canonical.actor) != frozen_actor
        assert int(actual.updates) == step + 1
        assert int(actual.actor.step) == 0 and int(actual.q1.step) == step + 1
        for key in metrics:
            np.testing.assert_array_equal(metrics[key], expected_metrics[key])
        if step == 0:
            assert serialization.to_bytes(state) == original
        state = actual


@pytest.mark.parametrize("teacher", [False, True])
def test_rejects_bc_or_teacher_state(example, teacher):
    state, batch = example
    if teacher:
        batch["teacher_accepted"][0] = True
    else:
        state = state.replace(bc_updates=1)
    with pytest.raises(ValueError, match="BC/teacher0"):
        critic_evaluation_step(state, batch, 6000)
