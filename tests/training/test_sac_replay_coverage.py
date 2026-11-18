"""Coverage sampling statistics reproduce the production replay RNG exactly."""

import numpy as np

from scripts.audit_sac_replay_coverage import sampling_exposure
from src.training.sac.replay import Replay


def test_exposure_matches_actual_replay_sample_and_final_rng():
    replay = Replay(capacity=7, seed=13)
    for array in replay.arrays.values():
        array[...] = 0
    for i in range(7):
        replay.arrays["raw_u"][i] = i
    actual_counts = np.zeros(7, np.int64)
    actual_fresh = 0
    for size in range(4, 8):
        replay.size = size
        draw = replay.sample(256)["raw_u"][:, 0, 0].astype(int)
        actual_counts += np.bincount(draw, minlength=7)
        actual_fresh += int(np.count_nonzero(draw >= 3))
    counts, state, fresh = sampling_exposure(13, 4, warmup=3)
    np.testing.assert_array_equal(counts, actual_counts)
    assert counts.sum() == 4 * 256
    assert state == replay.rng.bit_generator.state
    assert fresh == actual_fresh
