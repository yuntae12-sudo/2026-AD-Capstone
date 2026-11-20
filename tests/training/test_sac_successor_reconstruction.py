from copy import deepcopy

import pytest

from scripts.sac_successor_reconstruction import successor_history_indices


def records():
    def row(tick, kind, **lineage):
        return dict(pre_tick=tick, scene_id="scene", provider="fixed_keep", episode_uid="episode",
                    bootstrap=True, lineage=dict(kind=kind, **lineage))
    root = dict(source_collection_id=2, source_stage="nonterminal", refresh_path_id=1)
    # Source action at row1 is replaced by row2; row3 belongs to the original environment.
    return [row(0, "main"), row(1, "main"),
            row(1, "successor_refresh", **root, refresh_path_step=0, previous_refresh_collection_id=None),
            row(2, "main"),
            row(2, "successor_refresh", **root, refresh_path_step=1, previous_refresh_collection_id=3)]


def test_actual_successor_chain_excludes_original_source_and_interleaved_main():
    rows = records()
    assert successor_history_indices(rows, 4) == [0, 2, 4]
    assert successor_history_indices(rows, 2) == [0, 2]


@pytest.mark.parametrize("mutation", ["wrong_predecessor", "cross_terminal", "scene", "path", "source", "terminal_stage"])
def test_reject_broken_or_crossed_path(mutation):
    rows = deepcopy(records())
    if mutation == "wrong_predecessor":
        rows[4]["lineage"]["previous_refresh_collection_id"] = 4
    elif mutation == "cross_terminal":
        rows[2]["bootstrap"] = False
    elif mutation == "scene":
        rows[4]["scene_id"] = "another"
    elif mutation == "path":
        rows[4]["lineage"]["refresh_path_id"] = 3
    elif mutation == "source":
        rows[4]["lineage"]["source_collection_id"] = 1
    else:
        rows[4]["lineage"]["source_stage"] = "terminal"
    with pytest.raises(ValueError):
        successor_history_indices(rows, 4)
