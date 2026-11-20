"""Protect both command sources without changing successor execution or actions.

Four nonterminal slots remain fully usable by a sole observed command. When
both commands arrive, evict the oldest entry of a command holding over two
slots. Terminal retention, round-robin draws, and one actual active path are
inherited unchanged. This is an opt-in collection comparison, not adoption.
"""
from collections import Counter
import hashlib
from pathlib import Path

from scripts.sac_rejection_refresh import RejectionRefresh
from scripts.sac_successor_refresh import SuccessorRefresh, refresh_spec as successor_spec


PROVIDERS = ("fixed_keep", "fixed_follow")


def refresh_spec():
    original = successor_spec()
    return dict(original, mode="actual_provider_protected_successor_refresh_v1",
                nonterminal_eviction="oldest_of_command_with_more_than_two_slots",
                nonterminal_command_reservation=2,
                nonterminal_unused_slots_borrowable=True,
                supported_providers=list(PROVIDERS),
                successor_script_sha256=original["script_sha256"],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class ProtectedNonterminalBank(RejectionRefresh):
    def after_base(self, snapshot, boundary, step, **identity):
        if identity["provider"] not in PROVIDERS:
            raise ValueError("provider protection requires fixed KEEP or FOLLOW")
        super().after_base(snapshot, boundary, step, **identity)
        # Terminal additions are never allowed to displace nonterminal sources.
        self.entries = type(self.entries)(e for e in self.entries if e.outcome is None)
        if len(self.entries) > 4:
            counts = Counter(e.provider for e in self.entries)
            index = next(i for i, e in enumerate(self.entries) if counts[e.provider] > 2)
            del self.entries[index]
            if index < self.cursor:
                self.cursor -= 1
        live = {e.collection_id for e in self.entries}
        self.source_checks = {i: c for i, c in self.source_checks.items() if i in live}


class ProviderProtectedSuccessorRefresh(SuccessorRefresh):
    def __init__(self):
        super().__init__()
        self.banks["nonterminal"] = ProtectedNonterminalBank()

    def after_base(self, snapshot, boundary, step, **identity):
        # Reject before either bank can mutate.
        if identity["provider"] not in PROVIDERS:
            raise ValueError("provider protection requires fixed KEEP or FOLLOW")
        return super().after_base(snapshot, boundary, step, **identity)
