import pytest

from scripts.diagnose_sac_branch_q import select_cases


def row(provider, reason, kind="failure_branch", terminal=True):
    return dict(provider=provider, terminal_reason=reason, lineage=dict(kind=kind),
                terminated=terminal, truncated=False)


def test_cases_selected_before_q_and_separated_by_provider():
    rows = [row("fixed_keep", "COLLISION", "main"),
            row("fixed_keep", None, terminal=False),
            row("fixed_keep", "COLLISION"), row("fixed_keep", "COLLISION"),
            row("fixed_follow", "OFFROAD")]
    assert select_cases(rows) == {"fixed_keep": 2, "fixed_follow": 4}


def test_missing_failure_is_not_treated_as_a_successful_probe():
    with pytest.raises(ValueError, match="no actual"):
        select_cases([row("fixed_keep", "ROUTE_GOAL")])
