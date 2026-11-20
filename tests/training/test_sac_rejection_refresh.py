from types import SimpleNamespace as NS

import pytest

from scripts.sac_rejection_refresh import RejectionRefresh, refresh_spec
from src.contracts.observation import CoreObservation


def add(bank, i, *, valid=True, outcome=None, bootstrap=None, kind="main"):
    b = NS(token=NS(episode_uid=f"ep{i}", sim_tick=i), needs_sac_action=True,
           sac_observation=CoreObservation.empty())
    env = NS(token=b.token, ledger={"mutable": []}, _step_cache={})
    env._boundary = lambda: b
    done = outcome is not None
    step = NS(boundary=NS(token=NS(sim_tick=i + 1)),
              result=NS(pre_token=b.token, terminated=done, truncated=False,
                        terminal_reason=NS(value=outcome) if done else None,
                        check_result=NS(valid=valid, reason="OK" if valid else "KINEMATIC_LIMIT")),
              transition=NS(bootstrap=not done if bootstrap is None else bootstrap))
    bank.after_base(env, b, step, collection_id=i, scene_id=f"s{i}",
                    provider="fixed_follow", kind=kind)
    return env, b, step


def test_nonterminal_rejection_is_selectable_without_fabricating_terminal():
    bank = RejectionRefresh()
    env, boundary, step = add(bank, 1, valid=False)
    for i in (2, 3, 4):
        add(bank, i)
    fork, restored, lineage = bank.draw()
    assert restored is boundary and fork.token == boundary.token
    assert lineage["kind"] == "rejection_refresh" and lineage["source_outcome"] is None
    assert lineage["source_collection_id"] == 1 and lineage["source_check_valid"] is False
    assert lineage["source_check_reason"] == "KINEMATIC_LIMIT"
    assert step.transition.bootstrap and not step.result.terminated
    fork.ledger["mutable"].append("revisit")
    assert env.ledger["mutable"] == []
    assert bank.draw() is None


@pytest.mark.parametrize("outcome", [None, "ROUTE_GOAL", "TASK_LOST"])
def test_accepted_states_and_nonfailure_terminals_do_not_enter(outcome):
    bank = RejectionRefresh()
    for i in range(1, 5):
        add(bank, i, valid=outcome is None, outcome=outcome)
    assert bank.draw() is None and len(bank.entries) == 0


@pytest.mark.parametrize("outcome", ["COLLISION", "OFFROAD"])
def test_actual_failure_still_enters_even_if_proposal_passed(outcome):
    bank = RejectionRefresh()
    add(bank, 1, valid=True, outcome=outcome)
    for i in (2, 3, 4):
        add(bank, i)
    _, _, lineage = bank.draw()
    assert lineage["source_outcome"] == outcome and lineage["source_check_valid"] is True


def test_rejected_state_fifo_and_cadence_remain_bounded():
    bank = RejectionRefresh()
    selected = []
    for i in range(1, 21):
        add(bank, i, valid=False)
        result = bank.draw()
        if result:
            selected.append(result[2]["source_collection_id"])
        assert set(bank.source_checks) == {e.collection_id for e in bank.entries}
        assert len(bank.entries) <= 8
    assert selected == [1, 2, 5, 9, 13]
    assert bank.base_steps == 20 and bank.refresh_steps == 5
    assert refresh_spec()["base_interval"] == 4 and refresh_spec()["capacity"] == 8
    bank.verify_preserved()


def test_refresh_cannot_recursively_seed_bank_and_invalid_flags_are_rejected():
    bank = RejectionRefresh()
    with pytest.raises(ValueError, match="recursively"):
        add(bank, 1, valid=False, kind="rejection_refresh")
    with pytest.raises(ValueError, match="bootstrap"):
        add(bank, 1, valid=False, bootstrap=False)
    assert bank.base_steps == 0 and len(bank.entries) == 0
