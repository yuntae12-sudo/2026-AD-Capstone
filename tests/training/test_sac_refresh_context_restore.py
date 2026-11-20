"""Resume must reject cuts that silently discard unfinished collector work."""
from copy import deepcopy
import pytest

from scripts.sac_refresh_context_restore import validate_cut


def main_cut():
    report = dict(status="COMPLETE", inherited_online_updates=0,
        inherited_self_warmup=5000, updates=2, new_transitions=2,
        episodes=[dict(outcome="DIAGNOSTIC_BUDGET")],
        collection=dict(complete_path=True, samples=1, window=6,
                        selection="moving_predecessors", rollout_horizon=200))
    rows = [dict(collection_id=i, update=i, policy_updates_before=i-1,
                 delta_ticks=1, bootstrap=i == 1,
                 lineage=dict(kind="main" if i == 1 else "successor_refresh"))
            for i in (1, 2)]
    return report, rows


def test_completed_terminal_refresh_does_not_end_live_main_episode():
    report, rows = main_cut()
    validate_cut(report, rows)
    # A live main cut without an optional refresh is also actionable.
    report.update(updates=1, new_transitions=1)
    validate_cut(report, rows[:1])


@pytest.mark.parametrize("change", [
    "branch_cut", "terminated_main", "missing_update", "duplicate_id",
    "multiple_physical_ticks", "resumed_run", "unfinished_report", "wrong_branch_protocol",
])
def test_unsupported_or_corrupt_cut_fails_before_environment_restore(change):
    report, rows = deepcopy(main_cut())
    if change == "branch_cut":
        rows[0]["lineage"]["kind"] = "failure_branch"
    elif change == "terminated_main":
        rows[0]["bootstrap"] = False
    elif change == "missing_update":
        report["updates"] = 3
    elif change == "duplicate_id":
        rows[1]["collection_id"] = 1
    elif change == "multiple_physical_ticks":
        rows[0]["delta_ticks"] = 2
    elif change == "resumed_run":
        report["inherited_online_updates"] = 1
    elif change == "unfinished_report":
        report["status"] = "RUNNING"
    else:
        report["collection"]["complete_path"] = False
    with pytest.raises(ValueError):
        validate_cut(report, rows)
