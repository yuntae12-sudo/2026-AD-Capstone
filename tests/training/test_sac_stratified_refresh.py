from types import SimpleNamespace as NS

import pytest

from scripts.sac_stratified_refresh import StratifiedRefresh, refresh_spec
from src.contracts.observation import CoreObservation


def add(bank, i, *, outcome=None, valid=False, bootstrap=None, kind="main"):
    boundary = NS(token=NS(episode_uid=f"ep{i}", sim_tick=i),
                  needs_sac_action=True, sac_observation=CoreObservation.empty())
    environment = NS(token=boundary.token, ledger={"mutable": []}, _step_cache={})
    environment._boundary = lambda: boundary
    terminal = outcome is not None
    step = NS(boundary=NS(token=NS(sim_tick=i + 1)),
              result=NS(pre_token=boundary.token, terminated=terminal, truncated=False,
                        terminal_reason=NS(value=outcome) if terminal else None,
                        check_result=NS(valid=valid, reason="OK" if valid else "KINEMATIC_LIMIT")),
              transition=NS(bootstrap=not terminal if bootstrap is None else bootstrap))
    bank.after_base(environment, boundary, step, collection_id=i,
                    scene_id=f"s{i}", provider="fixed_keep", kind=kind)
    return environment, boundary, step


def test_terminal_survives_continuous_rejections_and_draws_alternate():
    bank = StratifiedRefresh()
    original, _, terminal_step = add(bank, 1, outcome="COLLISION")
    selections = []
    for i in range(2, 33):
        add(bank, i)
        result = bank.draw()
        if result:
            environment, _, lineage = result
            selections.append((lineage["source_stage"], lineage["source_collection_id"]))
            environment.ledger["mutable"].append("new actual visit")
            assert bank.draw() is None
        assert len(bank.entries) <= 8
    assert [s for s, _ in selections] == ["terminal", "nonterminal"] * 4
    assert [i for s, i in selections if s == "terminal"] == [1] * 4
    assert original.ledger["mutable"] == [] and not terminal_step.transition.bootstrap
    assert bank.base_steps == 32 and bank.refresh_steps == 8
    bank.verify_preserved()


@pytest.mark.parametrize("stage", ["terminal", "nonterminal"])
def test_single_available_stage_uses_cadence_without_accumulating_credit(stage):
    bank = StratifiedRefresh()
    for i in range(1, 9):
        add(bank, i, valid=True)
        assert bank.draw() is None
    draws = []
    for i in range(9, 29):
        add(bank, i, outcome="OFFROAD" if stage == "terminal" else None)
        result = bank.draw()
        if result:
            lineage = result[2]
            assert lineage["source_stage"] == stage
            draws.append(lineage["source_collection_id"])
    assert draws == [9, 13, 17, 21, 25]
    assert bank.refresh_steps == 5 and len(bank.entries) == 4


def test_fifo_is_independent_and_total_memory_stays_eight():
    bank = StratifiedRefresh()
    for i in range(1, 21):
        add(bank, i, outcome="COLLISION" if i % 2 else None)
    assert [e.collection_id for e in bank.banks["terminal"].entries] == [13, 15, 17, 19]
    assert [e.collection_id for e in bank.banks["nonterminal"].entries] == [14, 16, 18, 20]
    assert len(bank.entries) == 8
    assert refresh_spec()["capacity"] == 8 and refresh_spec()["capacity_per_stage"] == 4


@pytest.mark.parametrize("outcome", ["ROUTE_GOAL", "TASK_LOST"])
def test_nonfailure_terminal_is_not_admitted_even_when_rejected(outcome):
    bank = StratifiedRefresh()
    for i in range(1, 5):
        add(bank, i, outcome=outcome)
    assert not bank.entries and bank.draw() is None


def test_recursive_sources_invalid_flags_and_mutated_snapshot_are_rejected():
    bank = StratifiedRefresh()
    with pytest.raises(ValueError, match="recursively"):
        add(bank, 1, kind="stratified_refresh")
    with pytest.raises(ValueError, match="bootstrap"):
        add(bank, 1, bootstrap=False)
    assert bank.base_steps == 0 and not bank.entries
    original, _, _ = add(bank, 1, outcome="COLLISION")
    for i in range(2, 5):
        add(bank, i)
    original.token = NS(episode_uid="mutated", sim_tick=1)
    with pytest.raises(RuntimeError, match="mutated"):
        bank.draw()
