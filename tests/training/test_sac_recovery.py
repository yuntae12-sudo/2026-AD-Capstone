from pathlib import Path
import numpy as np
import pytest
from src.contracts.observation import CoreObservation
from src.training.sac.replay import Replay, Transition
from src.training.sac.runtime import (
    save_collection_boundary,
    restore_collection_boundary,
)


def test_collection_resume_preserves_rng_cursor_and_archives_uncommitted_trace(
    tmp_path,
):
    versions = {"test": 1}
    replay = Replay(4, versions=versions)
    obs = CoreObservation.empty()
    replay.add(
        Transition(
            obs,
            np.zeros((30, 2)),
            0.25,
            obs,
            1,
            False,
            True,
            True,
            "episode",
            "scene",
            0,
            True,
        )
    )
    rng = np.random.default_rng(22)
    trace = tmp_path / "ticks.jsonl"
    trace.write_bytes(b'{"committed":true}\n')
    report = dict(episodes=[{"done": True}], seed=0, target=20000, elapsed_s=10.0)
    save_collection_boundary(tmp_path, replay, report, rng, [2, 0, 1], 1, 0)
    with trace.open("ab") as f:
        f.write(b'{"interrupted":true}\n')
    restored, state, next_rng = restore_collection_boundary(tmp_path, versions)
    assert (
        restored.size == 1 and state["next_index"] == 1 and state["order"] == [2, 0, 1]
    )
    np.testing.assert_array_equal(
        rng.integers(100, size=10), next_rng.integers(100, size=10)
    )
    assert trace.read_bytes() == b'{"committed":true}\n'
    assert (
        next(tmp_path.glob("uncheckpointed-*.jsonl")).read_bytes()
        == b'{"interrupted":true}\n'
    )
    np.testing.assert_array_equal(
        restored.arrays["raw_u"][:1], replay.arrays["raw_u"][:1]
    )
    with pytest.raises(ValueError, match="incompatible"):
        restore_collection_boundary(tmp_path, {"test": 2})


def test_recovery_retains_two_complete_boundaries(tmp_path):
    replay = Replay(2, versions={})
    rng = np.random.default_rng(0)
    for count in (1, 2, 3):
        save_collection_boundary(
            tmp_path, replay, dict(episodes=[{}] * count), rng, [0], 1, count
        )
    assert sorted(p.name for p in (tmp_path / "recovery").iterdir()) == [
        "boundary-000002",
        "boundary-000003",
    ]
    loaded, state, _ = restore_collection_boundary(tmp_path, {})
    assert state["cycles"] == 3 and loaded.size == 0
