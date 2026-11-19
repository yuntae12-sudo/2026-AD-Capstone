from types import SimpleNamespace as NS

import pytest

from src.contracts.observation import CoreObservation
from scripts.sac_failure_branches import FailureBranches


class Environment:
    def __init__(self, tick, speed=1.0):
        self.token = NS(episode_uid="parent", sim_tick=tick)
        self.ledger = {"events": []}
        self._step_cache = {}
        self.ego = NS(speed=speed)
        self.boundary = NS(token=self.token, needs_sac_action=True,
                           sac_observation=CoreObservation.empty())

    def _boundary(self):
        return self.boundary


def outcome(reason, terminal=True):
    return NS(terminated=terminal, truncated=False, terminal_reason=NS(value=reason),
              pre_token=NS(sim_tick=10))


def test_real_branches_isolate_parent_and_each_other_and_keep_lineage():
    buffer = FailureBranches(2, 2)
    parents = [Environment(t) for t in (8, 9, 10)]
    for parent in parents:
        buffer.remember(parent, parent.boundary)
    seen = []
    for branch, boundary, lineage in buffer.generate(outcome("COLLISION")):
        assert branch.ledger == {"events": []}
        assert not branch._step_cache
        branch.ledger["events"].append("actual step")
        branch._step_cache["runtime"] = object()
        assert boundary.token.episode_uid == "parent"
        assert lineage["parent_tick"] == boundary.token.sim_tick
        seen.append((lineage["branch_sample"], lineage["parent_tick"]))
    assert seen == [(0, 10), (0, 9), (1, 10), (1, 9)]
    assert all(p.ledger == {"events": []} and not p._step_cache for p in parents)


def test_no_success_corruption_or_nonterminal_trigger_and_opt_in_only():
    buffer = FailureBranches(3, 2)
    parent = Environment(10)
    buffer.remember(parent, parent.boundary)
    for reason in ("ROUTE_GOAL", "DATA_CORRUPTION", "TIME_LIMIT", "RECORD_END"):
        assert not list(buffer.generate(outcome(reason)))
    assert not list(buffer.generate(outcome("COLLISION", terminal=False)))
    disabled = FailureBranches()
    disabled.remember(parent, parent.boundary)
    assert not list(disabled.generate(outcome("COLLISION")))


def test_stale_boundary_and_invalid_counts_fail_before_collection():
    for samples, window in ((-1, 2), (65, 2), (2, 0), (2, 33)):
        with pytest.raises(ValueError):
            FailureBranches(samples, window)
    parent = Environment(10)
    with pytest.raises(ValueError, match="current policy"):
        FailureBranches(1).remember(parent, Environment(9).boundary)


def test_moving_selection_keeps_earliest_rejections_and_terminal_with_same_budget():
    buffer = FailureBranches(2, 3, "moving_rejections")
    for tick in range(11):
        parent = Environment(tick, speed=0.0 if tick == 1 else 1.0)
        buffer.remember(parent, parent.boundary)
        result = outcome("COLLISION")
        result.pre_token = parent.token
        result.check_result = NS(valid=tick not in (1, 2, 4, 7, 10))
        buffer.after_step(result)
    branches = list(buffer.generate(outcome("COLLISION")))
    assert [lineage["parent_tick"] for _, _, lineage in branches] == [10, 4, 2, 10, 4, 2]
    assert [lineage["selection_role"] for _, _, lineage in branches[:3]] == ["terminal", "moving_rejection", "moving_rejection"]


def test_moving_selection_fills_missing_early_states_without_duplicate_ticks():
    buffer = FailureBranches(1, 3, "moving_rejections")
    for tick in range(11):
        parent = Environment(tick, speed=0.0)
        buffer.remember(parent, parent.boundary)
        result = outcome("COLLISION")
        result.pre_token = parent.token
        result.check_result = NS(valid=False)
        buffer.after_step(result)
    branches = list(buffer.generate(outcome("COLLISION")))
    assert [lineage["parent_tick"] for _, _, lineage in branches] == [10, 9, 8]
    with pytest.raises(ValueError, match="selection"):
        FailureBranches(1, 3, "unknown")


def collect_selected_history(window, accepted, moving, samples=1):
    buffer = FailureBranches(samples, window, "moving_predecessors")
    for tick in range(11):
        parent = Environment(tick, speed=1.0 if tick in moving else 0.0)
        buffer.remember(parent, parent.boundary)
        result = outcome("COLLISION")
        result.pre_token = parent.token
        result.check_result = NS(valid=tick in accepted)
        buffer.after_step(result)
    return list(buffer.generate(outcome("COLLISION")))


def test_predecessors_preserve_valid_moving_history_and_terminal_budget():
    branches = collect_selected_history(6, set(range(5)), set(range(6)), samples=2)
    first = [meta for _, _, meta in branches[:6]]
    assert [m["parent_tick"] for m in first] == [10, 4, 3, 2, 1, 0]
    assert first[0]["selection_role"] == "terminal"
    assert all(m["selection_role"] == "moving_predecessor" for m in first[1:])
    assert all(m["predecessor_rejection_tick"] == 5 for m in first[1:])
    assert len(branches) == 12
    assert len({(m["parent_tick"], m["branch_sample"]) for _, _, m in branches}) == 12


def test_predecessors_fill_with_rejections_without_duplicate_parent_states():
    branches = collect_selected_history(6, {0, 1, 4}, set(range(6)))
    selected = {m["parent_tick"]: m for _, _, m in branches}
    assert len(branches) == len(selected) == 6
    assert set(selected) == {0, 1, 2, 3, 4, 10}
    assert selected[0]["predecessor_rejection_tick"] == 2
    assert selected[1]["predecessor_rejection_tick"] == 2
    assert selected[4]["predecessor_rejection_tick"] == 5
    assert selected[2]["selection_role"] == selected[3]["selection_role"] == "moving_rejection"


def test_stopped_and_stale_accepted_states_are_not_moving_predecessors():
    # Tick0 is accepted/moving but too old; tick4 is recent/accepted but stopped.
    branches = collect_selected_history(3, {0, 4}, {0, 5})
    assert [m["parent_tick"] for _, _, m in branches] == [10, 9, 5]
    assert all(m["selection_role"] != "moving_predecessor" for _, _, m in branches)
    # A one-state window reserves its only state for the actual terminal step.
    one = collect_selected_history(1, set(range(5)), set(range(6)), samples=2)
    assert [m["parent_tick"] for _, _, m in one] == [10, 10]
