import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.sac_provider_protected_refresh import ProviderProtectedSuccessorRefresh, refresh_spec
from scripts.sac_successor_refresh import SuccessorRefresh
from src.contracts.observation import CoreObservation


def base(bank, provider="fixed_keep", *, terminal=False, valid=False):
    i = bank.last_actual_collection_id + 1
    before = NS(token=NS(episode_uid=f"episode{i}", sim_tick=0),
                needs_sac_action=True, sac_observation=CoreObservation.empty())
    env = NS(token=before.token, ledger={}, _step_cache={}, _boundary=lambda: before)
    step = NS(boundary=NS(token=NS(episode_uid=before.token.episode_uid, sim_tick=1)),
              result=NS(pre_token=before.token, terminated=terminal, truncated=False,
                        terminal_reason=NS(value="COLLISION") if terminal else None,
                        check_result=NS(valid=valid, reason="OK" if valid else "KINEMATIC_LIMIT")),
              transition=NS(bootstrap=not terminal))
    bank.after_base(env, before, step, collection_id=i, scene_id=f"scene{i}",
                    provider=provider, kind="main")


def execute(bank, draw, terminal=False):
    env, before, _ = draw
    after = NS(token=NS(episode_uid=before.token.episode_uid, sim_tick=before.token.sim_tick+1),
               needs_sac_action=not terminal, sac_observation=CoreObservation.empty())
    env.token = after.token
    env._boundary = lambda: after
    step = NS(boundary=after, result=NS(pre_token=before.token, terminated=terminal,
                                       truncated=False), transition=NS(bootstrap=not terminal))
    bank.after_refresh(step, collection_id=bank.last_actual_collection_id+1)


def ids(bank, stage="nonterminal"):
    return [e.collection_id for e in bank.banks[stage].entries]


@pytest.mark.parametrize("provider", ["fixed_keep", "fixed_follow"])
def test_single_command_preserves_full_capacity_and_historical_choices(provider):
    old, new = SuccessorRefresh(), ProviderProtectedSuccessorRefresh()
    for cycle in range(6):
        for bank in (old, new):
            for _ in range(4):
                base(bank, provider, terminal=(cycle % 2 == 1))
        assert ids(old) == ids(new) and ids(old, "terminal") == ids(new, "terminal")
        a, b = old.draw(), new.draw()
        assert a[2] == b[2]
        execute(old, a, terminal=True)
        execute(new, b, terminal=True)
    assert len(new.entries) == 8


@pytest.mark.parametrize("rare,frequent", [("fixed_follow", "fixed_keep"), ("fixed_keep", "fixed_follow")])
def test_two_rare_sources_survive_many_other_command_arrivals(rare, frequent):
    bank = ProviderProtectedSuccessorRefresh()
    base(bank, rare); base(bank, rare)
    for _ in range(40):
        base(bank, frequent)
        assert {1, 2} <= set(ids(bank))
        assert len(ids(bank)) <= 4
    assert ids(bank) == [1, 2, 41, 42]
    assert Counter(e.provider for e in bank.banks["nonterminal"].entries) == {rare: 2, frequent: 2}
    assert set(bank.banks["nonterminal"].source_checks) == set(ids(bank))


def test_new_command_borrows_only_until_two_sources_exist():
    bank = ProviderProtectedSuccessorRefresh()
    for _ in range(4): base(bank)
    assert ids(bank) == [1, 2, 3, 4]
    base(bank, "fixed_follow")
    assert ids(bank) == [2, 3, 4, 5]
    base(bank)
    assert ids(bank) == [3, 4, 5, 6]
    base(bank, "fixed_follow")
    assert ids(bank) == [4, 5, 6, 7]


def test_terminal_arrivals_cannot_displace_saved_nonterminal_sources():
    bank = ProviderProtectedSuccessorRefresh()
    for p in ("fixed_keep", "fixed_keep", "fixed_follow", "fixed_follow"): base(bank, p)
    for _ in range(12): base(bank, terminal=True)
    assert ids(bank) == [1, 2, 3, 4] and ids(bank, "terminal") == [13, 14, 15, 16]
    assert len(bank.entries) == 8


def test_active_keep_finishes_then_retained_follow_is_actually_drawn():
    bank = ProviderProtectedSuccessorRefresh()
    for _ in range(4): base(bank)
    first = bank.draw(); execute(bank, first)
    for p in ("fixed_follow", "fixed_follow", "fixed_keep", "fixed_keep"): base(bank, p)
    second = bank.draw()
    assert second[0] is first[0] and second[2]["refresh_path_step"] == 1
    execute(bank, second, terminal=True)
    for _ in range(4): base(bank)
    third = bank.draw()
    assert third[2]["provider"] == "fixed_follow" and third[2]["refresh_path_step"] == 0
    execute(bank, third)
    assert bank.active.lineage["provider"] == "fixed_follow"
    assert bank.draw() is None and bank.refresh_steps == 3 and bank.base_steps == 12


def test_unknown_provider_rejected_before_either_bank_mutates():
    bank = ProviderProtectedSuccessorRefresh()
    with pytest.raises(ValueError, match="fixed KEEP or FOLLOW"):
        base(bank, "unsupported", terminal=True)
    assert bank.base_steps == 0 and not bank.entries and bank.last_actual_collection_id == 0


def test_spec_preserves_capacity_original_actions_and_one_active_path():
    spec = refresh_spec()
    assert spec["capacity_per_stage"] == 4 and spec["max_retained_environments"] == 10
    assert spec["active_paths"] == 1 and spec["base_interval"] == 4
    assert spec["action"] == "original_current_gaussian" and spec["recursive_refresh"] is False


def test_training_body_diff_is_limited_to_collector_import_and_provenance():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    old = (scripts / "diagnose_sac_successor_refresh.py").read_text()
    new = (scripts / "diagnose_sac_provider_protected_refresh.py").read_text()
    new = new.replace("from scripts.sac_provider_protected_refresh import ProviderProtectedSuccessorRefresh as SuccessorRefresh, refresh_spec",
                      "from scripts.sac_successor_refresh import SuccessorRefresh, refresh_spec")
    new = new.replace("OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON", "OWN_SUCCESSOR_REFRESH_SAC_COMPARISON")
    new = new.replace("provider_protected_refresh=refresh_spec()", "successor_refresh=refresh_spec()")
    a, b = ast.parse(old), ast.parse(new)
    a.body.pop(0); b.body.pop(0)  # Only the module documentation differs additionally.
    assert ast.dump(a) == ast.dump(b)
