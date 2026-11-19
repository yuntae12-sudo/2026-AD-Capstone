from types import SimpleNamespace as NS

import pytest

from scripts.sac_failure_rollouts import collect_failure_rollouts
from src.contracts.observation import CoreObservation


def boundary(tick):
    return NS(token=NS(episode_uid="same-parent", sim_tick=tick),
              needs_sac_action=True, sac_observation=CoreObservation.empty())


class Roots:
    samples = 1
    window = 2

    def generate(self, result):
        if result != "COLLISION":
            return
        for root_tick in (10, 4):
            b = boundary(root_tick)
            e = NS(token=b.token, root_tick=root_tick)
            yield e, b, dict(kind="failure_branch", branch_sample=0,
                            parent_episode_uid="same-parent", parent_tick=root_tick,
                            observation_sha256="root-fingerprint")


def callback(rows, *, terminate_root=None):
    def train(environment, before, lineage):
        after = boundary(before.token.sim_tick + 1)
        environment.token = after.token
        terminal = environment.root_tick == terminate_root
        after.needs_sac_action = not terminal
        result = NS(pre_token=before.token, terminated=terminal, truncated=False)
        transition = NS(bootstrap=not terminal)
        rows.append((lineage, before.token.sim_tick, transition, result))
        return NS(boundary=after, result=result, transition=transition), len(rows)
    return train


def test_round_robin_actual_successors_and_lineage_not_root_restarts():
    rows = []
    count = collect_failure_rollouts(Roots(), "COLLISION", horizon=3,
                                    train_step=callback(rows), can_step=lambda: True)
    assert count == 6
    assert [r[1] for r in rows] == [10, 4, 11, 5, 12, 6]
    assert [r[0]["parent_tick"] for r in rows] == [10, 4, 10, 4, 10, 4]
    assert [r[0]["previous_collection_id"] for r in rows] == [None, None, 1, 2, 3, 4]
    assert [r[0]["branch_step"] for r in rows] == [0, 0, 1, 1, 2, 2]
    assert all(r[2].bootstrap and not r[3].terminated for r in rows)


def test_actual_terminal_stops_only_that_branch_and_budget_does_not_truncate():
    rows = []
    count = collect_failure_rollouts(Roots(), "COLLISION", horizon=8,
                                    train_step=callback(rows, terminate_root=10),
                                    can_step=lambda: len(rows) < 4)
    assert count == 4
    assert [r[1] for r in rows] == [10, 4, 5, 6]
    assert not rows[0][2].bootstrap
    assert rows[-1][2].bootstrap and not rows[-1][3].truncated
    assert [r[0]["previous_collection_id"] for r in rows] == [None, None, 2, 3]


def test_no_success_trigger_or_empty_budget_and_no_fake_callback_steps():
    rows = []
    for result, budget in [("ROUTE_GOAL", True), ("COLLISION", False)]:
        assert collect_failure_rollouts(Roots(), result, horizon=8,
                                       train_step=callback(rows), can_step=lambda: budget) == 0
    assert not rows

    def broken(environment, b, lineage):
        return NS(boundary=b, result=NS(pre_token=b.token), transition=NS()), 1
    with pytest.raises(RuntimeError, match="exactly one actual step"):
        collect_failure_rollouts(Roots(), "COLLISION", horizon=8,
                                train_step=broken, can_step=lambda: True)


def test_reject_unbounded_or_mixed_sampling_before_execution():
    for horizon in (0, 1, 9, 1.5):
        with pytest.raises(ValueError, match="horizon"):
            collect_failure_rollouts(Roots(), "COLLISION", horizon=horizon,
                                    train_step=None, can_step=None)
    roots = Roots()
    roots.samples = 8
    with pytest.raises(ValueError, match="one path"):
        collect_failure_rollouts(roots, "COLLISION", horizon=8,
                                train_step=None, can_step=None)
