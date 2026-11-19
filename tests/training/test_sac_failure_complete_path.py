from types import SimpleNamespace as NS

import pytest

from scripts.sac_failure_complete_path import collect_complete_failure_path
from scripts.diagnose_sac_q_values import observation_fingerprint
from src.contracts.observation import CoreObservation


def boundary(tick):
    return NS(token=NS(episode_uid="main", sim_tick=tick), needs_sac_action=True,
              sac_observation=CoreObservation.empty())


class Roots:
    samples, window, selection = 1, 6, "moving_predecessors"

    def __init__(self, states=((30, "terminal"), (4, "moving_predecessor"),
                               (2, "moving_predecessor"), (1, "moving_rejection"))):
        self.states = states

    def generate(self, result):
        if result != "COLLISION":
            return
        for tick, role in self.states:
            b = boundary(tick)
            env = NS(token=b.token)
            yield env, b, dict(kind="failure_branch", parent_tick=tick,
                              branch_sample=0, selection_role=role,
                              observation_sha256=observation_fingerprint(b.sac_observation))


def callback(rows, end, truncated=False):
    def train(env, before, lineage):
        # Real callback stand-in: the collector never creates or modifies flags.
        done = before.token.sim_tick == end
        after = boundary(before.token.sim_tick + 1)
        after.needs_sac_action = not done
        env.token = after.token
        step = NS(boundary=after, transition=NS(bootstrap=not done),
                  result=NS(pre_token=before.token, terminated=done and not truncated,
                            truncated=done and truncated))
        rows.append((lineage, before.token.sim_tick, step))
        return step, len(rows)
    return train


@pytest.mark.parametrize("truncated", [False, True])
def test_earliest_accepted_predecessor_continues_beyond_eight_and_stops_at_real_terminal(truncated):
    rows = []
    n = collect_complete_failure_path(Roots(), "COLLISION", horizon=200,
        train_step=callback(rows, 17, truncated), can_step=lambda: True)
    assert n == 16 and [r[1] for r in rows] == list(range(2, 18))
    assert all(r[0]["parent_tick"] == 2 and r[0]["branch_sample"] == 0 for r in rows)
    assert [r[0]["branch_step"] for r in rows] == list(range(16))
    assert [r[0]["previous_collection_id"] for r in rows] == [None, *range(1, 16)]
    assert all(r[2].transition.bootstrap for r in rows[:-1])
    assert not rows[-1][2].transition.bootstrap
    assert rows[-1][2].result.truncated == truncated


def test_global_budget_keeps_nonterminal_bootstrap_and_fallback_uses_earliest_moving_rejection():
    rows = []
    roots = Roots(((30, "terminal"), (9, "recent"), (4, "moving_rejection"), (3, "moving_rejection")))
    n = collect_complete_failure_path(roots, "COLLISION", horizon=200,
        train_step=callback(rows, 20), can_step=lambda: len(rows) < 10)
    assert n == 10 and [r[1] for r in rows] == list(range(3, 13))
    assert all(r[2].transition.bootstrap for r in rows)
    assert all(not r[2].result.terminated and not r[2].result.truncated for r in rows)


def test_no_failure_and_invalid_modes_do_not_execute():
    assert collect_complete_failure_path(Roots(), "ROUTE_GOAL", horizon=200,
        train_step=None, can_step=lambda: True) == 0
    with pytest.raises(ValueError, match="complete path"):
        collect_complete_failure_path(Roots(), "COLLISION", horizon=8, train_step=None, can_step=None)
    roots = Roots(); roots.samples = 8
    with pytest.raises(ValueError, match="complete path"):
        collect_complete_failure_path(roots, "COLLISION", horizon=200, train_step=None, can_step=None)


def test_horizon_overrun_is_error_not_an_artificial_terminal():
    rows = []
    with pytest.raises(RuntimeError, match="did not reach terminal"):
        collect_complete_failure_path(Roots(), "COLLISION", horizon=200,
            train_step=callback(rows, 300), can_step=lambda: True)
    assert len(rows) == 200 and all(r[2].transition.bootstrap for r in rows)
