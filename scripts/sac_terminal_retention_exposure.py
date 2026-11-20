"""Reconstruct actual goal/failure replay exposure without changing sampling."""
import numpy as np

from scripts.sac_terminal_retention_sampling import draw_indices
from scripts.sac_diagnostic_sampling import indices_hash


def sampling_exposure(seed, updates, bootstrap, reward, *, warmup=5000, draw_log):
    if type(updates) is not int or updates <= 0 or type(warmup) is not int or warmup <= 0:
        raise ValueError("positive update and warmup counts required")
    if len(draw_log) != updates:
        raise ValueError("complete draw history required")
    if len(bootstrap) < warmup + updates or len(reward) < warmup + updates:
        raise ValueError("complete actual replay prefix required")
    counts = np.zeros(warmup + updates, np.int64)
    rng = np.random.default_rng(seed)
    latest = 0
    for i, row in enumerate(draw_log, start=1):
        size = warmup + i
        indices = draw_indices(rng, size, bootstrap, reward)
        if (row.get("update") != i or row.get("replay_size") != size
                or row.get("replay_indices_sha256") != indices_hash(indices)):
            raise ValueError(f"terminal replay draw history mismatch at update {i}")
        np.add.at(counts, indices, 1)
        latest += int(np.count_nonzero(indices >= max(warmup, size - 100)))
    return counts, rng.bit_generator.state, latest
