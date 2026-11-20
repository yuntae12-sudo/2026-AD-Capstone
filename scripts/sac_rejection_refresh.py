"""Actual-policy revisits of rejected nonterminal and terminal failure states.

Only bank eligibility changes from TerminalRefresh. The caller must execute the
original stochastic policy/collector and one full SAC update per new transition.
Source outcomes are metadata; they never determine a revisit's reward or flags.
"""
import hashlib
from pathlib import Path

from scripts.sac_terminal_refresh import Entry, TerminalRefresh, refresh_spec as terminal_spec


def refresh_spec():
    original = terminal_spec()
    return dict(original, mode="actual_rejection_refresh_v1",
                source_outcomes=[None, "COLLISION", "OFFROAD"],
                eligibility="rejected_nonterminal_or_actual_collision_offroad",
                original_bank_sha256=original["script_sha256"],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class RejectionRefresh(TerminalRefresh):
    """Keep FIFO8, round-robin, base4 cadence and nonrecursive sources intact."""

    def __init__(self):
        super().__init__()
        self.source_checks = {}

    def after_base(self, snapshot, boundary, step, **identity):
        valid = step.result.check_result.valid
        reason = step.result.check_result.reason
        # Reuse actual-step, source identity, bootstrap and snapshot validation;
        # this also inserts actual COLLISION/OFFROAD pre-states as before.
        super().after_base(snapshot, boundary, step, **identity)
        terminal = step.result.terminated or step.result.truncated
        collection_id = identity["collection_id"]
        if not terminal and not valid:
            if len(self.entries) == 8:
                self.entries.popleft()
                self.cursor = max(0, self.cursor - 1)
            from scripts.diagnose_sac_q_values import observation_fingerprint
            self.entries.append(Entry(snapshot, boundary, collection_id,
                                      identity["scene_id"], identity["provider"],
                                      None, observation_fingerprint(boundary.sac_observation)))
        live_ids = {entry.collection_id for entry in self.entries}
        self.source_checks = {i: check for i, check in self.source_checks.items() if i in live_ids}
        if collection_id in live_ids:
            self.source_checks[collection_id] = dict(source_check_valid=bool(valid),
                                                     source_check_reason=reason)

    def draw(self):
        result = super().draw()
        if result is None:
            return None
        environment, boundary, lineage = result
        lineage.update(kind="rejection_refresh",
                       **self.source_checks[lineage["source_collection_id"]])
        return environment, boundary, lineage
