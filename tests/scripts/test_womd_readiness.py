"""Dataset partitioning must not leak test scenes or double-count episodes."""

import json
from types import SimpleNamespace
import pytest

from scripts.screen_womd_training import development_split, prefilter, summarize
from scripts import screen_womd_training
from scripts.benchmark_womd_control import timing
from src.environment.womd_control.environment import SACPlanningEnvironment
from src.environment.womd_control.geometry import ResetRejected
from tests.environment.test_womd_control import scene


@pytest.mark.parametrize(
    "scene_id,expected",
    [
        ("d675f8a070ebe2e4", ("train", 5)),
        ("5fe0296e76253bda", ("validation", 82)),
        ("91c716a16256612a", ("test", 91)),
    ],
)
def test_split_v2_known_assignments(scene_id, expected):
    assert development_split(scene_id) == expected


def test_missing_identity_cannot_be_assigned_to_train():
    with pytest.raises(ValueError):
        development_split("")


def test_screen_does_not_inspect_or_cache_test_scene_geometry(tmp_path, monkeypatch):
    proto = scene()
    proto.scenario_id = "91c716a16256612a"
    root, output = tmp_path / "source", tmp_path / "output"
    root.mkdir()
    (root / "training_20s.tfrecord-00000-of-01000").write_bytes(b"unit fixture")
    monkeypatch.setattr(
        screen_womd_training, "raw_records", lambda _: [proto.SerializeToString()]
    )
    monkeypatch.setattr(
        screen_womd_training,
        "prefilter",
        lambda _: pytest.fail("test geometry must remain held out"),
    )
    screen_womd_training.screen(
        SimpleNamespace(root=root, output=output, start_shard=0, shards=1)
    )
    report = json.loads((output / "screen-00000.json").read_text())
    assert report["rows"][0]["status"] == "HELD_OUT"
    assert not list((output / "candidates").iterdir())


def test_prefilter_uses_same_rounded_anchor_and_route_as_real_reset():
    proto = scene()
    proto.tracks[0].states[2].center_x += 0.00000013
    result = prefilter(proto)
    e = SACPlanningEnvironment()
    e.reset(proto)
    assert result["route_lane_ids"] == e.route.lane_ids
    assert result["boundary_feature_ids"] == e.route.boundary_ids
    assert result["start_frame"] == e.start_frame
    assert result["initial_speed_mps"] == e.ego.speed
    proto.timestamps_seconds[4] += 0.02
    for check in (prefilter, SACPlanningEnvironment().reset):
        with pytest.raises(ResetRejected, match="TIMESTEP"):
            check(proto)


def test_duplicate_ids_are_quarantined_not_counted_as_more_data(tmp_path):
    candidate = dict(scene_id="duplicate", split="train", status="ROUTE_CANDIDATE")
    (tmp_path / "screen-00000.json").write_text(
        json.dumps(dict(elapsed_s=1, rows=[candidate, candidate]))
    )
    reset = dict(
        candidate,
        reset_results=[
            dict(
                status="RESET_ACCEPTED",
                provider="fixed_keep",
                remaining_ticks_upper_bound=197,
            )
        ],
    )
    (tmp_path / "reset-00000-00000.json").write_text(json.dumps(reset))
    summarize(tmp_path)
    assert json.loads((tmp_path / "eligible_manifest.json").read_text())["rows"] == []
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["duplicate_scene_ids"] == ["duplicate"]
    assert report["teacher_transitions_collected"] == 0
    assert not report["independent_groups_verified"]


def test_timing_keeps_empty_warmup_and_unequal_samples_explicit():
    assert timing([])["mean_s"] is None
    result = timing([1.0, 3.0])
    assert result["ticks_per_s"] == 0.5
    assert result["mean_s"] == 2.0
    assert result["p95_s"] == pytest.approx(2.9)
