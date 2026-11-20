"""Event denominators and actual selected-path tracking for saved captures."""
from copy import deepcopy
import math

import pytest

from scripts.sac_capture_metrics import episode_metrics, summarize


def trace(scene="a"):
    return dict(scene_id=scene, provider="fixed_follow", outcome="COLLISION", frames=[
        dict(tick=0, outcome=None, ego=dict(speed=2.0, xy=[0., 0.])),
        dict(tick=1, proposal_tick=0, outcome=None, sample_valid=True,
             accepted=False, source="REUSED", events=[], reward=-.01,
             ego=dict(speed=1., xy=[3., 4.]), executed_xy=[[0., 0.]],
             raw_xy=[[3., 4.]]),
        dict(tick=2, proposal_tick=1, outcome="COLLISION", sample_valid=True,
             accepted=False, source="EMERGENCY", events=["COLLISION", "OFFROAD"],
             reward=-20., ego=dict(speed=0., xy=[3., 4.]), executed_xy=[], raw_xy=[]),
    ])


def test_tracking_uses_reused_path_and_excludes_emergency():
    e = episode_metrics(trace())
    assert e["tracking_samples"] == 1
    assert e["tracking_position_rms"] == 5.0  # Raw proposal would give zero.
    assert e["tracking_position_squared"] == 25.0
    assert e["first_braking_tick"] == 2
    s = summarize([e])
    assert s["collision_rate"] == s["offroad_rate"] == 1.0
    assert s["raw_rejection_rate"] == 1.0
    assert s["braking_intervention_rate"] == s["stationary_fraction"] == .5


def test_pooled_rms_weights_actual_tracking_samples():
    a, b = trace(), trace("b")
    b["outcome"] = b["frames"][-1]["outcome"] = "ROUTE_GOAL"
    b["frames"][-1].update(source="BRAKE", events=[], executed_xy=[[3., 4.]])
    b["frames"][1].update(source="NEW", accepted=True, executed_xy=[[3., 4.]])
    values = [episode_metrics(x) for x in (a, b)]
    s = summarize(values)
    assert s["tracking_samples"] == 3
    assert s["tracking_position_rms"] == pytest.approx(math.sqrt(25 / 3))
    assert s["success_rate"] == s["collision_rate"] == s["offroad_rate"] == .5


def test_no_tracking_samples_is_none_not_zero():
    t = trace()
    t["frames"][1].update(source="EMERGENCY", executed_xy=[])
    e = episode_metrics(t)
    assert e["tracking_samples"] == 0 and e["tracking_position_rms"] is None
    assert summarize([e])["tracking_position_rms"] is None


@pytest.mark.parametrize("change", [
    lambda t: t.update(outcome="PREVIEW_LIMIT"),
    lambda t: t["frames"][-1].update(outcome="DATA_CORRUPTION"),
    lambda t: t["frames"][1].update(sample_valid=False),
    lambda t: t["frames"][1].update(tick=4),
    lambda t: t["frames"][1].update(proposal_tick=1),
    lambda t: t["frames"][1].update(executed_xy=[]),
    lambda t: t["frames"][-1].update(executed_xy=[[0., 0.]]),
    lambda t: t["frames"][1].update(reward=float("nan")),
    lambda t: t["frames"][1]["ego"].update(speed=float("inf")),
    lambda t: t["frames"][-1].update(events=["COLLISION", "COLLISION"]),
])
def test_incomplete_or_inconsistent_capture_is_rejected(change):
    t = deepcopy(trace())
    change(t)
    with pytest.raises(ValueError):
        episode_metrics(t)


def test_duplicate_pairs_do_not_inflate_denominator():
    e = episode_metrics(trace())
    with pytest.raises(ValueError):
        summarize([e, e])
    with pytest.raises(ValueError):
        summarize([])
