from types import SimpleNamespace as NS

import pytest

from src.contracts.observation import CoreObservation
from scripts.sac_failure_branches import FailureBranches


class Environment:
    def __init__(self, tick):
        self.token = NS(episode_uid="parent", sim_tick=tick)
        self.ledger = {"events": []}
        self._step_cache = {}
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
