from types import SimpleNamespace as NS

import pytest

from scripts.diagnose_sac_q_values import observation_fingerprint
from scripts.sac_failure_rollout_budget import collect_fixed_budget_rollouts
from scripts.sac_failure_quota_extension import collect_quota_then_extension
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
    samples, window, selection = 8, 6, "moving_predecessors"

    def generate(self, result):
        if result != "COLLISION":
            return
        for tick, role in ((30, "terminal"), (4, "moving_predecessor"),
                           (2, "moving_predecessor")):
            env = Environment(tick)
            yield env, env.boundary, dict(
                parent_episode_uid="parent", parent_tick=tick, branch_sample=0,
                selection_role=role,
                observation_sha256=observation_fingerprint(env.boundary.sac_observation))


def callback(rows, end_after=20, truncated=False):
    def train(env, before, lineage):
        env.ledger.append(before.token.sim_tick)
        done = (env.root_tick == 30 or before.token.sim_tick == env.root_tick +
                (1 if env.root_tick == 4 else end_after - 1))
        after = boundary(before.token.sim_tick + 1)
        after.needs_sac_action = not done
        env.token, env.boundary = after.token, after
        step = NS(boundary=after, transition=NS(bootstrap=not done),
                  result=NS(pre_token=before.token, terminated=done and not truncated,
                            truncated=done and truncated))
        rows.append((dict(lineage), before.token.sim_tick, not done, list(env.ledger)))
        return step, len(rows)
    return train


@pytest.mark.parametrize("truncated", [False, True])
def test_original_quota_prefix_is_identical_then_selected_open_path_reaches_real_terminal(truncated):
    original, extended = [], []
    collect_fixed_budget_rollouts(Roots(), "COLLISION", horizon=8,
        train_step=callback(original, truncated=truncated), can_step=lambda: True)
    count = collect_quota_then_extension(Roots(), "COLLISION", horizon=8,
        train_step=callback(extended, truncated=truncated), can_step=lambda: True)
    assert count == 36 and len(original) == 24
    assert extended[:24] == original
    suffix = extended[24:]
    assert [r[1] for r in suffix] == list(range(10, 22))
    assert [r[0]["root_budget_step"] for r in suffix] == list(range(8, 20))
    assert [r[0]["previous_collection_id"] for r in suffix] == list(range(24, 36))
    assert all(r[0]["parent_tick"] == 2 and r[0]["branch_sample"] == 0 for r in suffix)
    assert all(r[0]["quota_extension"] for r in suffix)
    assert suffix[-1][3] == list(range(2, 22))
    assert all(r[2] for r in suffix[:-1]) and not suffix[-1][2]


@pytest.mark.parametrize("budget", [11, 24, 28])
def test_budget_cut_preserves_actual_bootstrap_and_does_not_skip_quota(budget):
    rows = []
    count = collect_quota_then_extension(Roots(), "COLLISION", horizon=8,
        train_step=callback(rows), can_step=lambda: len(rows) < budget)
    assert count == budget
    assert all(not r[0].get("quota_extension", False) for r in rows[:24])
    if budget >= 24:
        assert rows[-1][2]  # No manufactured terminal at global budget cutoff.


def test_selected_terminal_last_quota_step_is_not_restarted_for_extension():
    rows = []
    count = collect_quota_then_extension(Roots(), "COLLISION", horizon=8,
        train_step=callback(rows, end_after=8), can_step=lambda: True)
    assert count == 24 and not rows[-1][2]
    assert all(not r[0].get("quota_extension", False) for r in rows)


def test_extension_finishes_current_path_after_original_quota_restart():
    rows = []
    count = collect_quota_then_extension(Roots(), "COLLISION", horizon=8,
        train_step=callback(rows, end_after=6), can_step=lambda: True)
    assert count == 28
    assert [r[0]["branch_sample"] for r in rows[24:]] == [1]*4
    assert [r[0]["branch_step"] for r in rows[24:]] == [2, 3, 4, 5]
    assert not rows[-1][2]


def test_nonfailure_invalid_mode_and_missing_actual_terminal():
    assert collect_quota_then_extension(Roots(), "ROUTE_GOAL", horizon=8,
        train_step=None, can_step=lambda: True) == 0
    with pytest.raises(ValueError, match="quota extension"):
        collect_quota_then_extension(Roots(), "COLLISION", horizon=7,
            train_step=None, can_step=None)
    rows = []
    with pytest.raises(RuntimeError, match="did not reach terminal"):
        collect_quota_then_extension(Roots(), "COLLISION", horizon=8,
            train_step=callback(rows, end_after=300), can_step=lambda: True)
    assert len(rows) == 216 and all(r[2] for r in rows[24:])
