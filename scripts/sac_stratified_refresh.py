"""Preserve terminal and nonterminal failure sources within eight total slots.

This opt-in bank changes state selection only. Actions, actual outcomes and full
SAC updates remain the caller's responsibility. Existing collectors stay frozen.
"""
import hashlib
from pathlib import Path

from scripts.sac_rejection_refresh import RejectionRefresh, refresh_spec as rejection_spec


def refresh_spec():
    original = rejection_spec()
    return dict(original, mode="actual_stratified_refresh_v1",
                capacity_per_stage=4,
                selection="alternate_available_stages_round_robin_fifo_within_stage",
                first_preferred_stage="terminal",
                terminal_bank_sha256=original["original_bank_sha256"],
                original_bank_sha256=original["script_sha256"],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class StratifiedRefresh:
    """Two bounded banks; a frequent rejection cannot evict a terminal source."""

    def __init__(self):
        self.banks = {stage: RejectionRefresh() for stage in ("terminal", "nonterminal")}
        self.next_stage = "terminal"
        self.refresh_steps = 0
        self.last_draw_base = 0

    @property
    def base_steps(self):
        return self.banks["terminal"].base_steps

    @property
    def entries(self):
        return tuple(entry for bank in self.banks.values() for entry in bank.entries)

    def after_base(self, snapshot, boundary, step, **identity):
        # Each validator sees the same actual step. At most four entries precede
        # insertion, so the inherited capacity-eight eviction cannot run.
        for stage, bank in self.banks.items():
            bank.after_base(snapshot, boundary, step, **identity)
            keep_terminal = stage == "terminal"
            bank.entries = type(bank.entries)(
                e for e in bank.entries if (e.outcome is not None) == keep_terminal)
            if len(bank.entries) > 4:
                bank.entries.popleft()
                bank.cursor = max(0, bank.cursor - 1)
            live = {e.collection_id for e in bank.entries}
            bank.source_checks = {i: c for i, c in bank.source_checks.items() if i in live}
        assert self.banks["nonterminal"].base_steps == self.base_steps

    def draw(self):
        if (not self.base_steps or self.base_steps % 4
                or self.last_draw_base == self.base_steps):
            return None
        other = "nonterminal" if self.next_stage == "terminal" else "terminal"
        stage = self.next_stage if self.banks[self.next_stage].entries else other
        if not self.banks[stage].entries:
            return None
        result = self.banks[stage].draw()
        assert result is not None
        environment, boundary, lineage = result
        self.refresh_steps += 1
        self.last_draw_base = self.base_steps
        self.next_stage = "nonterminal" if stage == "terminal" else "terminal"
        lineage.update(kind="stratified_refresh", source_stage=stage,
                       refresh_index=self.refresh_steps,
                       stage_bank_collection_ids={
                           name: [e.collection_id for e in bank.entries]
                           for name, bank in self.banks.items()})
        return environment, boundary, lineage

    def verify_preserved(self):
        for bank in self.banks.values():
            bank.verify_preserved()
