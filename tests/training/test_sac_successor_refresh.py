from types import SimpleNamespace as NS

import pytest

from scripts.sac_successor_refresh import SuccessorRefresh, refresh_spec
from src.contracts.observation import CoreObservation


def base(bank, *, terminal=False, valid=False, kind="main"):
    i = bank.last_actual_collection_id + 1
    boundary = NS(token=NS(episode_uid=f"episode{i}", sim_tick=0),
                  needs_sac_action=True, sac_observation=CoreObservation.empty())
    env = NS(token=boundary.token, ledger={"values": []}, _step_cache={})
    env._boundary = lambda: boundary
    step = NS(boundary=NS(token=NS(episode_uid=boundary.token.episode_uid, sim_tick=1)),
              result=NS(pre_token=boundary.token, terminated=terminal, truncated=False,
                        terminal_reason=NS(value="COLLISION") if terminal else None,
                        check_result=NS(valid=valid, reason="OK" if valid else "KINEMATIC_LIMIT")),
              transition=NS(bootstrap=not terminal))
    bank.after_base(env, boundary, step, collection_id=i, scene_id=f"scene{i}",
                    provider="fixed_keep", kind=kind)
    return env


def execute(bank, draw, *, terminal=False):
    env, before, _ = draw
    boundary = NS(token=NS(episode_uid=before.token.episode_uid, sim_tick=before.token.sim_tick+1),
                  needs_sac_action=not terminal, sac_observation=CoreObservation.empty())
    env.token = boundary.token
    env._boundary = lambda: boundary
    step = NS(boundary=boundary, result=NS(pre_token=before.token, terminated=terminal,
                                         truncated=False), transition=NS(bootstrap=not terminal))
    bank.after_refresh(step, collection_id=bank.last_actual_collection_id+1)
    return step


def advance(bank, **kwargs):
    for _ in range(4):
        base(bank, **kwargs)
    return bank.draw()


def test_successor_reuses_actual_environment_after_root_eviction_and_terminal_interleaving():
    bank = SuccessorRefresh()
    first = advance(bank)
    root_id = first[2]["source_collection_id"]
    execute(bank, first)
    previous_id = bank.last_actual_collection_id
    terminal = advance(bank, terminal=True)
    assert terminal[2]["source_stage"] == "terminal"
    execute(bank, terminal, terminal=True)
    follow = advance(bank)
    assert root_id not in follow[2]["bank_collection_ids"]
    assert follow[0] is first[0] and follow[1].token.sim_tick == 1
    assert follow[2]["source_collection_id"] == root_id
    assert follow[2]["previous_refresh_collection_id"] == previous_id
    assert follow[2]["refresh_path_step"] == 1
    execute(bank, follow, terminal=True)
    assert bank.active is None and bank.refresh_steps == 3 and bank.base_steps == 12
    terminal = advance(bank)
    assert terminal[2]["source_stage"] == "terminal"
    execute(bank, terminal, terminal=True)
    fresh = advance(bank)
    assert fresh[2]["refresh_path_step"] == 0 and fresh[0] is not first[0]
    assert len(bank.entries) <= 8 and refresh_spec()["active_paths"] == 1


def test_nonterminal_only_continues_without_recursive_insertion_or_extra_credit():
    bank = SuccessorRefresh()
    first = advance(bank)
    for i in range(3):
        assert first[2]["refresh_path_step"] == i
        count = len(bank.entries)
        execute(bank, first)
        assert len(bank.entries) == count and bank.draw() is None
        if i != 2:
            first = advance(bank, valid=True)
    assert bank.active.boundary.token.sim_tick == 3
    assert bank.refresh_steps * 4 == bank.base_steps


def test_terminal_source_new_nonterminal_outcome_is_not_forced_terminal_or_continued():
    bank = SuccessorRefresh()
    first = advance(bank, terminal=True)
    result = execute(bank, first, terminal=False)
    assert result.transition.bootstrap and bank.active is None
    second = advance(bank, terminal=True)
    assert second[2]["refresh_path_step"] == 0
    assert second[0] is not first[0]


def test_pending_outcome_required_and_invalid_flags_fail_closed():
    bank = SuccessorRefresh()
    draw = advance(bank)
    with pytest.raises(RuntimeError, match="recorded"):
        bank.draw()
    with pytest.raises(RuntimeError, match="recorded"):
        base(bank)
    env, before, _ = draw
    after = NS(token=NS(episode_uid=before.token.episode_uid, sim_tick=1),
               sac_observation=CoreObservation.empty())
    env.token = after.token
    env._boundary = lambda: after
    step = NS(boundary=after, result=NS(pre_token=before.token, terminated=True, truncated=False),
              transition=NS(bootstrap=True))
    with pytest.raises(ValueError, match="bootstrap"):
        bank.after_refresh(step, collection_id=bank.last_actual_collection_id+1)
    assert bank.pending is not None


def test_waiting_successor_and_evicted_root_are_protected():
    bank = SuccessorRefresh()
    draw = advance(bank)
    execute(bank, draw)
    bank.active.environment.token = NS(episode_uid="changed", sim_tick=1)
    with pytest.raises(RuntimeError, match="waiting successor"):
        base(bank)
    bank.active.environment.token = bank.active.boundary.token
    for _ in range(8):
        base(bank)
    bank.active.root_environment.token = NS(episode_uid="changed", sim_tick=0)
    with pytest.raises(RuntimeError, match="saved root"):
        bank.verify_preserved()


def test_refresh_cannot_become_source_and_ids_include_refresh_steps():
    bank = SuccessorRefresh()
    with pytest.raises(ValueError, match="recursively"):
        base(bank, kind="successor_refresh")
    draw = advance(bank)
    execute(bank, draw)
    with pytest.raises(RuntimeError, match="no pending"):
        bank.after_refresh(NS(), collection_id=bank.last_actual_collection_id+1)
    assert bank.last_actual_collection_id == 5
    base(bank)
    assert bank.last_actual_collection_id == 6
