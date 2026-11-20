import copy
import numpy as np
import pytest
from scripts.sac_terminal_retention_sampling import draw_indices
from scripts.sac_diagnostic_sampling import indices_hash
from scripts.sac_terminal_retention_exposure import sampling_exposure


def fixture():
    bootstrap = np.array([False, False, True, True, False, True, False])
    reward = np.array([.1, -20., 0., 0., .1, 0., -20.])
    rng = np.random.default_rng(11)
    draws = [draw_indices(rng, size, bootstrap, reward) for size in range(4, 8)]
    logs = [dict(update=i, replay_size=3+i, replay_indices_sha256=indices_hash(ix))
            for i, ix in enumerate(draws, start=1)]
    return bootstrap, reward, draws, logs, rng.bit_generator.state


def test_reconstructs_actual_logged_batches_and_final_rng():
    b, r, draws, logs, expected_rng = fixture()
    counts, state, latest = sampling_exposure(11, 4, b, r, warmup=3, draw_log=logs)
    expected = np.bincount(np.concatenate(draws), minlength=7)
    np.testing.assert_array_equal(counts, expected)
    assert counts.sum() == 4*256 and state == expected_rng
    assert latest == sum(np.count_nonzero(ix >= 3) for ix in draws)


@pytest.mark.parametrize('key,value', [('update',9), ('replay_size',99), ('replay_indices_sha256','bad')])
def test_rejects_corrupted_history(key, value):
    b, r, _, logs, _ = fixture();logs=copy.deepcopy(logs);logs[1][key]=value
    with pytest.raises(ValueError, match='draw history mismatch'):
        sampling_exposure(11, 4, b, r, warmup=3, draw_log=logs)


def test_rejects_reward_classification_changed_after_collection():
    b, r, _, logs, _ = fixture();r[0]=-20.
    with pytest.raises(ValueError, match='draw history mismatch'):
        sampling_exposure(11, 4, b, r, warmup=3, draw_log=logs)


def test_ignores_unused_capacity_but_rejects_incomplete_prefix():
    b, r, _, logs, _ = fixture()
    x = sampling_exposure(11, 4, b, r, warmup=3, draw_log=logs)
    y = sampling_exposure(11, 4, np.r_[b,False], np.r_[r,np.nan], warmup=3, draw_log=logs)
    np.testing.assert_array_equal(x[0],y[0]);assert x[1:]==y[1:]
    with pytest.raises(ValueError,match='complete actual replay prefix'):
        sampling_exposure(11,4,b[:-1],r[:-1],warmup=3,draw_log=logs)
    with pytest.raises(ValueError,match='complete draw history'):
        sampling_exposure(11,4,b,r,warmup=3,draw_log=logs[:-1])
