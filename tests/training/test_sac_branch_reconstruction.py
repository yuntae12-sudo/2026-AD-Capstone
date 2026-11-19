from copy import deepcopy

import pytest

from scripts.sac_branch_reconstruction import branch_history_indices


def fixture():
    def row(tick, lineage, bootstrap=True):
        return dict(pre_tick=tick, episode_uid="p", scene_id="s", provider="fixed_keep",
                    bootstrap=bootstrap, lineage=lineage)
    rows = [row(t, dict(kind="main"), t != 3) for t in range(4)]
    root = dict(kind="failure_branch", parent_episode_uid="p", parent_tick=1,
                branch_sample=0, branch_step=0, previous_collection_id=None)
    rows += [row(1, root), row(2, dict(root, branch_step=1, previous_collection_id=5)),
             row(3, dict(root, branch_step=2, previous_collection_id=6), False)]
    return rows


def test_reconstruction_follows_actual_branch_not_same_tick_main():
    rows = fixture()
    assert branch_history_indices(rows, 6) == [0, 4, 5, 6]
    assert branch_history_indices(rows, 4) == [0, 4]
    # Historical independent one-step branch rows have no continuation fields.
    del rows[4]["lineage"]["branch_step"]
    del rows[4]["lineage"]["previous_collection_id"]
    assert branch_history_indices(rows, 4) == [0, 4]


@pytest.mark.parametrize("mutation", ["main_link", "future_link", "terminal", "different_root", "skip_depth"])
def test_corrupt_or_cross_branch_history_rejected(mutation):
    rows = deepcopy(fixture())
    if mutation == "main_link":rows[6]["lineage"]["previous_collection_id"] = 3
    if mutation == "future_link":rows[6]["lineage"]["previous_collection_id"] = 7
    if mutation == "terminal":rows[5]["bootstrap"] = False
    if mutation == "different_root":rows[5]["lineage"]["parent_tick"] = 0
    if mutation == "skip_depth":rows[5]["lineage"]["branch_step"] = 0
    with pytest.raises(ValueError):branch_history_indices(rows, 6)
