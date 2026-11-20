"""Recent diagnostics preserve stored data and reproducible sampling lineage."""

import copy

import numpy as np
import pytest

from scripts.sac_diagnostic_sampling import (
    draw_indices, indices_hash, report_sampling_mode, sample_recent, sampling_spec,
)
from scripts.audit_sac_replay_coverage import sampling_exposure
from src.training.sac.replay import Replay


def replay_fixture(size=300, seed=7):
    replay = Replay(capacity=310, seed=seed)
    for key, array in replay.arrays.items():
        array[...] = False if array.dtype == bool else 0
    for i in range(size):
        for key in ("obs", "next_obs", "raw_u", "reward", "tick"):
            replay.arrays[key][i] = i
        replay.arrays["fields"][i] = bool(i % 2)
        replay.arrays["items"][i] = bool(i % 2)
    replay.size = replay.cursor = size
    return replay


def test_uniform_indices_match_unchanged_production_sampler():
    replay = replay_fixture()
    rng = np.random.default_rng(7)
    expected = draw_indices(rng, replay.size)
    batch = replay.sample(256)
    np.testing.assert_array_equal(batch["raw_u"][:, 0, 0], expected)
    assert replay.rng.bit_generator.state == rng.bit_generator.state


def test_recent_batch_preserves_actions_masks_and_raw_dtypes():
    replay = replay_fixture()
    rng = np.random.default_rng(7)
    expected = np.concatenate((rng.choice(np.arange(300), 128, replace=True),
                               rng.choice(np.arange(44, 300), 128, replace=True)))
    batch, digest = sample_recent(replay)
    assert digest == indices_hash(expected)
    assert expected.shape == (256,) and min(expected[128:]) >= 44
    np.testing.assert_array_equal(batch["raw_u"][:, 0, 0], expected)
    np.testing.assert_array_equal(batch["reward"], expected)
    for key, value in batch["observation"]["features"].items():
        np.testing.assert_array_equal(value.reshape(256, -1)[:, 0], expected)
        mask = batch["observation"]["field_masks"][key]
        np.testing.assert_array_equal(mask.reshape(256, -1)[:, 0], expected % 2)
    assert batch["raw_u"].dtype == np.float64
    assert replay.rng.bit_generator.state == rng.bit_generator.state


def test_saved_rng_resumes_identical_batch_and_draw_audit(tmp_path):
    replay = replay_fixture(size=295)
    logs = []
    actual = np.zeros(300, np.int64)
    for step in range(1, 6):
        replay.size = replay.cursor = 295 + step
        replay.arrays["raw_u"][295 + step - 1] = 295 + step - 1
        batch, digest = sample_recent(replay)
        actual += np.bincount(batch["raw_u"][:, 0, 0].astype(int), minlength=300)
        logs.append(dict(update=step, replay_size=replay.size, replay_indices_sha256=digest))
    counts, state, _ = sampling_exposure(7, 5, warmup=295, mode="half_recent256", draw_log=logs)
    np.testing.assert_array_equal(counts, actual)
    assert state == replay.rng.bit_generator.state
    path = tmp_path / "replay.npz"
    replay.save(path)
    restored = Replay.load(path, {})
    a, ah = sample_recent(replay)
    b, bh = sample_recent(restored)
    assert ah == bh
    np.testing.assert_array_equal(a["raw_u"], b["raw_u"])
    assert replay.rng.bit_generator.state == restored.rng.bit_generator.state
    logs[-1]["replay_indices_sha256"] = "corrupt"
    with pytest.raises(ValueError, match="history mismatch"):
        sampling_exposure(7, 5, warmup=295, mode="half_recent256", draw_log=logs)


@pytest.mark.parametrize("size,cursor", [(0, 0), (310, 0), (300, 299)])
def test_invalid_ring_never_consumes_rng(size, cursor):
    replay = replay_fixture()
    replay.size, replay.cursor = size, cursor
    before = copy.deepcopy(replay.rng.bit_generator.state)
    with pytest.raises(ValueError, match="without wrap"):
        sample_recent(replay)
    assert before == replay.rng.bit_generator.state


def test_legacy_mode_and_changed_metadata_are_not_silent():
    assert report_sampling_mode({}) == "uniform"
    spec = sampling_spec("half_recent256")
    assert report_sampling_mode({"replay_sampling": spec}) == "half_recent256"
    spec["recent_window"] = 100
    with pytest.raises(ValueError, match="specification"):
        report_sampling_mode({"replay_sampling": spec})
    with pytest.raises(ValueError, match="complete draw history"):
        sampling_exposure(0, 1, mode="half_recent256")
    with pytest.raises(ValueError, match="unknown"):
        report_sampling_mode({"replay_sampling": {"mode": "typo"}})
