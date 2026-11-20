"""Bounded actual-policy revisits to previously executed failure pre-states.

The bank selects states, never actions. The caller uses the original stochastic
policy and collector, then performs one full SAC update per actual transition.
Refresh results do not recursively seed the bank. No outcome is copied.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path

from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint


def refresh_spec():
    return dict(mode="actual_terminal_refresh_v1", capacity=8, base_interval=4,
                selection="round_robin_fifo_eviction", source_kinds=["main", "failure_branch"],
                source_outcomes=["COLLISION", "OFFROAD"], recursive_refresh=False,
                action="original_current_gaussian", script_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest())


@dataclass
class Entry:
    environment: object
    boundary: object
    collection_id: int
    scene_id: str
    provider: str
    outcome: str
    fingerprint: str


class TerminalRefresh:
    def __init__(self):
        self.entries = deque()
        self.cursor = 0
        self.base_steps = 0
        self.refresh_steps = 0
        self.last_draw_base = 0
        self.last_collection_id = 0

    def after_base(self, snapshot, boundary, step, *, collection_id, scene_id, provider, kind):
        if kind not in ("main", "failure_branch"):
            raise ValueError("refresh results must not recursively enter the bank")
        if (type(collection_id) is not int or collection_id <= self.last_collection_id
                or snapshot.token != boundary.token or not boundary.needs_sac_action
                or step.result.pre_token != boundary.token or step.transition is None
                or step.boundary.token.sim_tick != boundary.token.sim_tick + 1):
            raise ValueError("bank source must be one actual step from its snapshot")
        terminal = step.result.terminated or step.result.truncated
        if step.transition.bootstrap != (not terminal):
            raise ValueError("actual terminal/bootstrap mismatch")
        fingerprint = observation_fingerprint(boundary.sac_observation)
        if observation_fingerprint(snapshot._boundary().sac_observation) != fingerprint:
            raise ValueError("source snapshot no longer matches actual pre-state")
        self.last_collection_id = collection_id
        self.base_steps += 1
        outcome = step.result.terminal_reason.value if step.result.terminal_reason else None
        if terminal and outcome in ("COLLISION", "OFFROAD"):
            if len(self.entries) == 8:
                self.entries.popleft()
                self.cursor = max(0, self.cursor - 1)
            self.entries.append(Entry(snapshot, boundary, collection_id, scene_id,
                                      provider, outcome, fingerprint))

    def draw(self):
        if (not self.entries or self.base_steps % 4 or self.base_steps == 0
                or self.last_draw_base == self.base_steps):
            return None
        entry = self.entries[self.cursor % len(self.entries)]
        self.cursor = (self.cursor + 1) % len(self.entries)
        self.last_draw_base = self.base_steps
        self.refresh_steps += 1
        if (entry.environment.token != entry.boundary.token or
                observation_fingerprint(entry.environment._boundary().sac_observation) != entry.fingerprint):
            raise RuntimeError("saved failure pre-state was mutated")
        return fork_environment(entry.environment), entry.boundary, dict(
            kind="terminal_refresh", source_collection_id=entry.collection_id,
            source_outcome=entry.outcome, scene_id=entry.scene_id, provider=entry.provider,
            source_observation_sha256=entry.fingerprint,
            base_steps=self.base_steps, refresh_index=self.refresh_steps,
            bank_collection_ids=[e.collection_id for e in self.entries])

    def verify_preserved(self):
        for entry in self.entries:
            if (entry.environment.token != entry.boundary.token or
                    observation_fingerprint(entry.environment._boundary().sac_observation) != entry.fingerprint):
                raise RuntimeError("refresh mutated a saved source state")
