from types import SimpleNamespace as NS

import pytest

from scripts.sac_failure_rollout_budget import collect_fixed_budget_rollouts
from scripts.diagnose_sac_q_values import observation_fingerprint
from src.contracts.observation import CoreObservation


def boundary(tick):
    return NS(token=NS(episode_uid="parent", sim_tick=tick), needs_sac_action=True,
              sac_observation=CoreObservation.empty())


class Environment:
    def __init__(self, tick):
        self.boundary = boundary(tick)
        self.token = self.boundary.token
        self.root_tick = tick
        self.ledger = []
        self._step_cache = {}

    def _boundary(self):
        return self.boundary


class Roots:
    samples = 8
    window = 2

    def generate(self, result):
        if result != "COLLISION":return
        for tick in (10, 4):
            e = Environment(tick)
            yield e, e.boundary, dict(kind="failure_branch", parent_episode_uid="parent",
                parent_tick=tick, branch_sample=0,
                observation_sha256=observation_fingerprint(e.boundary.sac_observation))


def callback(rows, terminal_after):
    def train(environment, before, lineage):
        environment.ledger.append(before.token.sim_tick)
        after = boundary(before.token.sim_tick + 1)
        terminal = terminal_after(environment, before)
        after.needs_sac_action = not terminal
        environment.token, environment.boundary = after.token, after
        result = NS(pre_token=before.token, terminated=terminal, truncated=False)
        transition = NS(bootstrap=not terminal)
        rows.append((lineage, before.token.sim_tick, transition, list(environment.ledger)))
        return NS(boundary=after, result=result, transition=transition), len(rows)
    return train


def test_terminal_root_keeps_eight_actual_steps_while_other_path_advances():
    rows = []
    count = collect_fixed_budget_rollouts(Roots(), "COLLISION", horizon=8,
        train_step=callback(rows, lambda e,b: e.root_tick == 10), can_step=lambda: True)
    assert count == 16
    a,b = rows[::2], rows[1::2]
    assert [r[1] for r in a] == [10]*8
    assert [r[1] for r in b] == list(range(4,12))
    assert [r[0]["branch_sample"] for r in a] == list(range(8))
    assert all(r[0]["branch_step"] == 0 and r[0]["previous_collection_id"] is None for r in a)
    assert [r[0]["root_budget_step"] for r in b] == list(range(8))
    assert [r[0]["previous_collection_id"] for r in b] == [None,2,4,6,8,10,12,14]
    assert all(not r[2].bootstrap and r[3] == [10] for r in a)
    assert b[-1][2].bootstrap and b[-1][3] == list(range(4,12))


def test_later_terminal_starts_a_new_path_without_fake_transition_or_shared_memory():
    rows = []
    collect_fixed_budget_rollouts(Roots(), "COLLISION", horizon=8,
        train_step=callback(rows, lambda e,b: b.token.sim_tick == e.root_tick+1),
        can_step=lambda: len(rows) < 11)
    a,b = rows[::2],rows[1::2]
    assert [r[1] for r in a] == [10,11,10,11,10,11]
    assert [r[0]["branch_sample"] for r in a] == [0,0,1,1,2,2]
    assert [r[0]["previous_collection_id"] for r in a] == [None,1,None,5,None,9]
    assert [r[3] for r in a] == [[10],[10,11]]*3
    assert b[-1][2].bootstrap  # Global budget does not add truncation.


def test_quota_validation_and_nonfailure_do_not_execute():
    roots=Roots()
    with pytest.raises(ValueError,match="quota"):
        collect_fixed_budget_rollouts(roots,"COLLISION",horizon=7,train_step=None,can_step=None)
    roots.window=7
    with pytest.raises(ValueError,match="quota"):
        collect_fixed_budget_rollouts(roots,"COLLISION",horizon=8,train_step=None,can_step=None)
    assert collect_fixed_budget_rollouts(Roots(),"ROUTE_GOAL",horizon=8,train_step=None,can_step=lambda:True)==0
