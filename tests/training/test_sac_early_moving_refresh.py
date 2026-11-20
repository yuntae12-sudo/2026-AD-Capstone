"""Synthetic collector tests; these do not establish real driving performance."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import numpy as np

from scripts.sac_early_moving_refresh import EarlyMovingSuccessorRefresh, refresh_spec
from src.contracts.observation import CoreObservation
from src.environment.womd_control.control import Ego
from tests.training.test_sac_provider_protected_refresh import execute, ids


def base(bank, provider="fixed_keep", *, episode="episode", tick=0, speed=8.,
         terminal=False, valid=False):
    i = bank.last_actual_collection_id + 1
    before = NS(token=NS(episode_uid=episode, sim_tick=tick),
                needs_sac_action=True, sac_observation=CoreObservation.empty())
    anchor = Ego(np.zeros(2), np.array([speed, 0.]), np.zeros(2), True, 0., 4.5, 2.)
    env = NS(token=before.token, ledger={}, _step_cache={}, _boundary=lambda: before,
             planning_anchor=lambda token: anchor)
    step = NS(boundary=NS(token=NS(episode_uid=episode, sim_tick=tick+1)),
              result=NS(pre_token=before.token, terminated=terminal, truncated=False,
                        terminal_reason=NS(value="COLLISION") if terminal else None,
                        check_result=NS(valid=valid, reason="OK" if valid else "KINEMATIC_LIMIT")),
              transition=NS(bootstrap=not terminal))
    bank.after_base(env, before, step, collection_id=i, scene_id=episode,
                    provider=provider, kind="main")
    return i


@pytest.mark.parametrize("provider", ["fixed_keep", "fixed_follow"])
def test_early_moving_anchor_survives_braking_and_stopped_arrivals(provider):
    bank = EarlyMovingSuccessorRefresh()
    early = base(bank, provider, tick=2, speed=10.)
    for tick in range(3, 80):
        base(bank, provider, tick=tick, speed=.213 if tick < 30 else 0.)
    assert early in ids(bank)
    draw = bank.draw()
    # 78 arrivals are not due; advance to the next base4 slot.
    assert draw is None
    for _ in range(2): base(bank, provider, tick=80, speed=0.)
    draw = bank.draw()
    assert draw[2]["source_collection_id"] == early
    assert draw[2]["source_speed_mps"] == 10.
    assert draw[2]["root_selection"] == "early_moving"
    assert len(bank.entries) == 4


def test_latest_moving_episode_replaces_anchor_and_earlier_branch_refines_it():
    bank = EarlyMovingSuccessorRefresh()
    base(bank, episode="old", tick=1)
    for tick in range(6): base(bank, episode="stopped", tick=tick, speed=0.)
    assert bank.banks["nonterminal"].anchors["fixed_keep"] == 1
    base(bank, episode="new", tick=8)
    earlier = base(bank, episode="new", tick=4)
    base(bank, episode="new", tick=20)
    assert bank.banks["nonterminal"].anchors["fixed_keep"] == earlier
    tie = base(bank, episode="new", tick=4)
    assert bank.banks["nonterminal"].anchors["fixed_keep"] == tie


def test_command_alternation_only_at_new_roots_and_actual_path_is_not_preempted():
    bank = EarlyMovingSuccessorRefresh()
    for p in ("fixed_keep", "fixed_keep", "fixed_follow", "fixed_follow"):
        base(bank, p)
    first = bank.draw()
    assert first[2]["provider"] == "fixed_keep"
    execute(bank, first)
    for tick in range(4): base(bank, "fixed_follow", tick=tick+10, speed=0.)
    second = bank.draw()
    assert second[0] is first[0] and second[2]["refresh_path_step"] == 1
    execute(bank, second, terminal=True)
    for _ in range(4): base(bank)
    third = bank.draw()
    assert third[2]["provider"] == "fixed_follow"
    assert third[2]["source_speed_mps"] == 8.
    execute(bank, third, terminal=True)
    for _ in range(4): base(bank)
    assert bank.draw()[2]["provider"] == "fixed_keep"


def test_stopped_only_fallback_is_latest_and_terminal_retention_is_unchanged():
    bank = EarlyMovingSuccessorRefresh()
    for _ in range(4): last = base(bank, speed=0.)
    draw = bank.draw()
    assert draw[2]["source_collection_id"] == last
    assert draw[2]["root_selection"] == "latest_fallback"
    execute(bank, draw, terminal=True)
    retained = ids(bank)
    for _ in range(8): base(bank, terminal=True)
    assert ids(bank) == retained
    assert ids(bank, "terminal") == [10, 11, 12, 13]
    assert bank.draw()[2]["source_stage"] == "terminal"


@pytest.mark.parametrize("speed", [float("nan"), float("inf")])
def test_nonfinite_motion_rejected_before_either_bank_mutates(speed):
    bank = EarlyMovingSuccessorRefresh()
    with pytest.raises(ValueError, match="finite actual"):
        base(bank, speed=speed)
    assert not bank.entries and bank.base_steps == bank.last_actual_collection_id == 0


def test_motion_threshold_and_valid_paths_do_not_create_rejected_anchors():
    bank = EarlyMovingSuccessorRefresh()
    base(bank, speed=10., valid=True)
    base(bank, speed=.099999)
    assert not bank.banks["nonterminal"].anchors
    base(bank, speed=.1)
    assert bank.banks["nonterminal"].anchors == {"fixed_keep": 3}


def test_capacity_is_bounded_under_command_and_episode_turnover():
    bank = EarlyMovingSuccessorRefresh()
    for i in range(160):
        base(bank, ("fixed_keep", "fixed_follow")[i % 2], episode=f"ep{i//8}",
             tick=i % 8, speed=0. if i % 3 else 5., terminal=i % 5 == 0)
        nt = bank.banks["nonterminal"]
        assert len(bank.entries) <= 8
        assert len(nt.speeds) == len(nt.source_checks) == len(nt.entries) <= 4
        assert len(nt.anchors) <= 2
        assert set(nt.anchors.values()) <= set(ids(bank))
    assert refresh_spec()["max_retained_environments"] == 10


def test_training_body_changes_only_collector_and_provenance():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    old = (scripts / "diagnose_sac_provider_protected_refresh.py").read_text()
    new = (scripts / "diagnose_sac_early_moving_refresh.py").read_text()
    new = new.replace("sac_early_moving_refresh", "sac_provider_protected_refresh")
    new = new.replace("EarlyMovingSuccessorRefresh", "ProviderProtectedSuccessorRefresh")
    new = new.replace("OWN_EARLY_MOVING_REFRESH_SAC_COMPARISON", "OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON")
    new = new.replace("early_moving_refresh=", "provider_protected_refresh=")
    a, b = ast.parse(old), ast.parse(new)
    assert ast.dump(a) == ast.dump(b)
