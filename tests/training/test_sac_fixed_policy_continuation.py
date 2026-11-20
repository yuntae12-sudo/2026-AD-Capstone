"""Budget boundaries must retain bootstrap and serialized sampling state."""

from copy import deepcopy
import pytest
from scripts.sac_fixed_policy_continuation import continuation_plan


@pytest.fixture
def records():
    rng = {"bit_generator": "PCG64", "state": {"state": 123, "inc": 57}}
    parent = dict(status="COMPLETE", scope="FIXED_POLICY_CRITIC_EVALUATION_NOT_SAC_TRAINING",
                  qualified=False, production_checkpoint=False, budget=1000, critic_steps=1000,
                  actual_transitions=1000, inherited_replay_size=6000, final_replay_size=7000,
                  actor_updates=0, temperature_updates=0, actor_and_temperature_bytes_unchanged=True,
                  original_inputs_unchanged=True, versions={"core": "example"}, final_replay_rng=rng,
                  collection_seed=20261120,
                  episodes=[dict(rollout=i, root_index=i % 2, policy_seed=20261120+i,
                                 ticks=45 if i < 21 else 55,
                                 outcome="COLLISION" if i < 21 else "DIAGNOSTIC_BUDGET") for i in range(22)])
    last = dict(collection_id=1000, rollout=21, branch_step=54, bootstrap=True,
                terminated=False, truncated=False, outcome=None)
    meta = dict(size=7000, cursor=7000, capacity=50000, versions={"core": "example"}, rng=deepcopy(rng))
    return parent, last, meta


def test_budget_handoff_preserves_data_and_does_not_reuse_rollout_seed(records):
    before = deepcopy(records)
    plan = continuation_plan(*records, total_steps=5000)
    assert records == before
    assert plan["new_steps"] == 4000 and plan["final_replay_size"] == 11000
    assert plan["first_new_rollout"] == 22 and plan["collection_seed"] + 22 == 20261142
    assert plan["inherited_boundary"]["bootstrap"] is True
    assert plan["inherited_boundary"]["outcome"] == "DIAGNOSTIC_BUDGET"
    assert plan["exact_simulator_resume"] is False
    plan["inherited_replay_rng"]["state"]["state"] = 0
    assert records == before


def test_true_terminal_at_exact_budget_remains_terminal(records):
    parent, last, meta = records
    parent["episodes"][-1]["outcome"] = last["outcome"] = "COLLISION"
    last.update(bootstrap=False, terminated=True)
    plan = continuation_plan(parent, last, meta, 2000)
    assert not plan["inherited_boundary"]["bootstrap"]
    assert plan["inherited_boundary"]["outcome"] == "COLLISION"


@pytest.mark.parametrize("total", [True, 1000, 5001, 10000, 5000.0])
def test_unbounded_or_empty_budget_is_rejected(records, total):
    with pytest.raises(ValueError):
        continuation_plan(*records, total)


@pytest.mark.parametrize("change", [
    lambda p, t, m: p.update(status="RUNNING"),
    lambda p, t, m: p.update(actor_updates=1),
    lambda p, t, m: p.update(critic_steps=999),
    lambda p, t, m: p.update(production_checkpoint=True),
    lambda p, t, m: m.update(cursor=0),
    lambda p, t, m: m.update(capacity=11000),
    lambda p, t, m: m["rng"]["state"].update(state=0),
    lambda p, t, m: t.update(terminated=True),
    lambda p, t, m: t.update(outcome="COLLISION"),
    lambda p, t, m: p["episodes"][-1].update(policy_seed=1),
])
def test_changed_state_or_fabricated_boundary_is_rejected(records, change):
    change(*records)
    with pytest.raises(ValueError):
        continuation_plan(*records, 5000)
