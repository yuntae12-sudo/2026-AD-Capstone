from types import SimpleNamespace as NS

import pytest

from scripts.sac_terminal_refresh import TerminalRefresh
from src.contracts.observation import CoreObservation


def source(i, outcome=None, bootstrap=None):
    b = NS(token=NS(episode_uid=f"ep{i}", sim_tick=i), needs_sac_action=True,
           sac_observation=CoreObservation.empty())
    env = NS(token=b.token, ledger={"mutable": []}, _step_cache={})
    env._boundary = lambda: b
    done = outcome is not None
    step = NS(boundary=NS(token=NS(sim_tick=i+1)),
              result=NS(pre_token=b.token, terminated=done, truncated=False,
                        terminal_reason=NS(value=outcome) if done else None),
              transition=NS(bootstrap=not done if bootstrap is None else bootstrap))
    return env, b, step


def add(bank, i, outcome=None, **kwargs):
    env, b, step = source(i, outcome)
    bank.after_base(env, b, step, collection_id=i, scene_id=f"s{i}",
                    provider="fixed_keep", kind="main", **kwargs)
    return env, b, step


def test_only_actual_failures_enter_bank_and_empty_intervals_do_not_accumulate_credit():
    bank = TerminalRefresh()
    for i in range(1, 5):
        add(bank, i, "ROUTE_GOAL" if i == 4 else None)
        assert bank.draw() is None
    add(bank, 5, "COLLISION")
    assert bank.draw() is None
    for i in (6, 7, 8):
        add(bank, i)
    env, b, lineage = bank.draw()
    assert lineage["source_collection_id"] == 5
    assert lineage["base_steps"] == 8 and lineage["refresh_index"] == 1
    assert env.token == b.token and bank.draw() is None


def test_round_robin_fifo_eviction_bounded_fraction_and_mutable_state_isolation():
    bank = TerminalRefresh()
    picked = []
    for i in range(1, 21):
        add(bank, i, "COLLISION")
        drawn = bank.draw()
        if drawn:
            env, _, metadata = drawn
            picked.append(metadata["source_collection_id"])
            env.ledger["mutable"].append("changed fork")
            assert all(not e.environment.ledger["mutable"] for e in bank.entries)
            assert metadata["bank_collection_ids"] == list(range(max(1, i-7), i+1))
    assert picked == [1, 2, 5, 9, 13]
    assert bank.base_steps == 20 and bank.refresh_steps == 5 and len(bank.entries) == 8
    bank.verify_preserved()


@pytest.mark.parametrize("kind", ["terminal_refresh", "teacher", "unknown"])
def test_refresh_or_foreign_source_cannot_recursively_enter_bank(kind):
    env, b, step = source(1, "COLLISION")
    with pytest.raises(ValueError, match="recursively"):
        TerminalRefresh().after_base(env, b, step, collection_id=1,
            scene_id="s", provider="fixed_keep", kind=kind)


def test_reject_fabricated_terminal_and_stale_collection_identity():
    bank = TerminalRefresh()
    env, b, step = source(1, "COLLISION", bootstrap=True)
    with pytest.raises(ValueError, match="bootstrap"):
        bank.after_base(env, b, step, collection_id=1, scene_id="s",
                        provider="fixed_keep", kind="main")
    add(bank, 1, "OFFROAD")
    with pytest.raises(ValueError, match="actual step"):
        add(bank, 1, "COLLISION")


def test_mutated_saved_snapshot_is_detected_before_reuse():
    bank = TerminalRefresh()
    env, _, _ = add(bank, 1, "COLLISION")
    for i in (2, 3, 4):
        add(bank, i)
    env.token = NS(episode_uid="different", sim_tick=1)
    with pytest.raises(RuntimeError, match="mutated"):
        bank.draw()
