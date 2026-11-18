"""Counterfactual Waymax branches must not contaminate the recorded history."""

import numpy as np
import pytest

from scripts.diagnose_sac_q_values import (
    fork_environment,
    probe_ticks,
    observation_fingerprint,
)
from src.environment.womd_control.environment import SACPlanningEnvironment
from src.training.sac.collector import collect_step
from src.training.sac.policy import proposal_from_action
from tests.environment.test_womd_control import scene


@pytest.mark.parametrize("rejected", [False, True])
def test_counterfactual_branches_preserve_parent_and_reproduce_actual_step(rejected):
    e = SACPlanningEnvironment()
    boundary = e.reset(scene())
    zero = lambda env, tok: proposal_from_action(env, tok, np.zeros((30, 2)))
    # Populate old_plan and PD memory, so a rejected branch exercises reuse.
    boundary = collect_step(e, boundary, zero).boundary
    token, snapshot, memory, plan, streak = (
        e.token,
        e._state,
        e.memory,
        e.old_plan,
        e.failure_streak,
    )
    raw = np.zeros((30, 2))
    if rejected:
        raw[:, 1] = np.where(np.arange(30) % 2, 0.1, -0.1)
    propose = lambda env, tok: proposal_from_action(env, tok, raw)
    branches = [fork_environment(e), fork_environment(e)]
    results = []
    for branch in branches:
        results.append(collect_step(branch, boundary, propose))
        assert e.token == token and e._state is snapshot and e.memory == memory
        assert e.old_plan is plan and e.failure_streak == streak
        assert branch.ledger is not e.ledger and branch._step_cache is not e._step_cache
    real = collect_step(e, boundary, propose)
    for branch, result in zip(branches, results):
        np.testing.assert_array_equal(branch.ego.position, e.ego.position)
        np.testing.assert_array_equal(branch.ego.velocity, e.ego.velocity)
        assert result.result.source == real.result.source
        assert result.result.check_result == real.result.check_result
        assert result.reward == real.reward
        assert observation_fingerprint(
            result.boundary.sac_observation
        ) == observation_fingerprint(real.boundary.sac_observation)
        assert branch.memory == e.memory and branch.failure_streak == e.failure_streak
        assert result.result.check_result.valid != rejected


def test_probe_schedule_includes_terminal_and_failure_transitions():
    frames = [
        dict(accepted=True, source="NEW", ego={"speed": 4}, proposal_tick=i - 1)
        for i in range(25)
    ]
    frames[4]["accepted"] = False
    frames[6]["source"] = "EMERGENCY"
    frames[9]["ego"]["speed"] = 0
    assert probe_ticks(frames) == [0, 3, 5, 8, 10, 20, 23]


def test_q_summary_separates_state_ranking_terminal_truth_and_mc_replicates():
    from scripts.summarize_sac_q_audit import summarize

    def candidate(q, accepted, terminal=False):
        reward = -20.0 if terminal else -0.01
        return dict(
            q1=q,
            q2=q,
            min_q=q,
            raw_u=np.zeros((30, 2)).tolist(),
            check=dict(valid=accepted, reason="OK" if accepted else "KINEMATIC_LIMIT"),
            source="NEW" if accepted else "EMERGENCY",
            reward=dict(training=reward),
            outcome="COLLISION" if terminal else None,
            bootstrap=not terminal,
            q_minus_target=q - reward,
            policy_sigma_distance_rms=0.0,
            next_observation_sha256="same",
            delta_ticks=1,
            actual_control={},
        )

    trials = [
        dict(seed=i, soft_return=value, outcome="COLLISION")
        for i, value in enumerate([-10.0, -12.0])
    ]
    report = dict(
        status="COMPLETE",
        episodes=[
            dict(
                status="COMPLETE",
                scene_id="a",
                provider="fixed_keep",
                reproduced_ticks=11,
                probes=[
                    dict(
                        tick=0,
                        speed=1.0,
                        candidates={
                            "mean": candidate(2.0, False),
                            "sample_0": candidate(1.0, True),
                        },
                        continuations={"mean": trials},
                    ),
                    dict(
                        tick=10,
                        speed=0.0,
                        candidates={
                            "mean": candidate(0.5, False, True),
                            "sample_0": candidate(-0.5, False, True),
                        },
                        continuations={},
                    ),
                ],
            )
        ],
    )
    summary, tables = summarize(report)
    assert summary["states"] == 2 and summary["candidates"] == 4
    assert summary["mixed_on_policy_states"] == 1
    assert summary["top_invalid_with_valid_alternative"] == 1
    assert summary["by_kind"]["mean"]["terminal_mean_abs_error"] == 20.5
    assert summary["by_kind"]["sample"]["terminal_mean_abs_error"] == 19.5
    assert summary["equivalent_rejected_groups"] == 1
    assert summary["largest_equivalent_rejected_q_span"] == 1.0
    assert (
        summary["continuation_runs"] == 2
        and summary["continuation_mean_q_error"] == 13.0
    )
    assert len(tables["rankings"]) == 2
    report["status"] = "RUNNING"
    with pytest.raises(ValueError, match="complete audit required"):
        summarize(report)
