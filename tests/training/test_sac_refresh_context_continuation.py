"""Prevent implicit mode changes and fresh/restarted learner mislabeling."""
from copy import deepcopy
import pytest

from scripts.continue_sac_refresh_context import validate_budget


def test_trained_checkpoint_restores_jit_mapping_order_and_scalar_types():
    import jax
    import numpy as np
    from flax import serialization, struct
    from scripts.continue_sac_refresh_context import restore_trained_learner

    @struct.dataclass
    class State:
        params: object
        updates: object
        bc_updates: int = 0

    template = State({'z': np.array([1.0], np.float32),
                      'a': np.array([2.0], np.float32)}, 0)
    with jax.default_device(jax.devices('cpu')[0]):
        trained = jax.jit(lambda s: s.replace(
            params=jax.tree_util.tree_map(lambda p: p + 1, s.params),
            updates=s.updates + 1))(template)
    # The real learner increments updates outside JIT and preserves bc_updates.
    trained = trained.replace(updates=1, bc_updates=0)
    weights = serialization.to_bytes(trained)
    assert serialization.to_bytes(serialization.from_bytes(template, weights)) != weights
    device = jax.devices('cpu')[0]
    restored = restore_trained_learner(template, weights, device)
    assert serialization.to_bytes(restored) == weights
    assert type(restored.updates) is int and type(restored.bc_updates) is int
    assert all(x.devices() == {device} for x in restored.params.values())
    np.testing.assert_array_equal(restored.params['z'], [2.0])


def test_trained_checkpoint_does_not_ignore_incompatible_template():
    import jax
    import numpy as np
    from flax import serialization
    from scripts.continue_sac_refresh_context import restore_trained_learner

    with pytest.raises(ValueError):
        restore_trained_learner({'required': np.ones(2)},
                                serialization.to_bytes({'wrong': np.ones(2)}),
                                jax.devices('cpu')[0])


def source():
    return dict(status="COMPLETE", scope="OWN_EARLY_MOVING_REFRESH_SAC_COMPARISON",
        updates=100, new_transitions=100, inherited_online_updates=0,
        inherited_self_warmup=5000, early_moving_refresh={"mode": "declared"})


def test_budget_is_total_not_additional_and_requires_an_unambiguous_source():
    assert validate_budget(source(), 120) == "early_moving_refresh"
    control = source()
    control["scope"] = "OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON"
    control["provider_protected_refresh"] = control.pop("early_moving_refresh")
    assert validate_budget(control, 10000) == "provider_protected_refresh"


@pytest.mark.parametrize("budget", [0, 99, 100, 10001, True, 120.0])
def test_invalid_or_noop_budget_rejected(budget):
    with pytest.raises(ValueError):
        validate_budget(source(), budget)


@pytest.mark.parametrize("change", ["running", "mixed_contract", "unknown_contract", "already_continued", "missing_actual_step"])
def test_invalid_provenance_rejected(change):
    s = deepcopy(source())
    if change == "running":
        s["status"] = "RUNNING"
    elif change == "mixed_contract":
        s["provider_protected_refresh"] = {}
    elif change == "unknown_contract":
        s["scope"] = "ANOTHER_EXPERIMENT"
    elif change == "already_continued":
        s["inherited_online_updates"] = 50
    else:
        s["new_transitions"] = 99
    with pytest.raises(ValueError):
        validate_budget(s, 120)
