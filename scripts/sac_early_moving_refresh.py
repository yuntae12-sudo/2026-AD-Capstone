"""Retain early moving failures and alternate commands at new actual path roots.

An opt-in collection comparison: rewards, actions, replay sampling and SAC
updates are untouched. Each command protects its earliest moving rejected
pre-state in the most recently encountered episode having such a state.
Stopped-only episodes do not erase that anchor. Active paths finish normally.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path

from scripts.sac_provider_protected_refresh import PROVIDERS, refresh_spec as protected_spec
from scripts.sac_rejection_refresh import RejectionRefresh
from scripts.sac_successor_refresh import SuccessorRefresh


def refresh_spec():
    original = protected_spec()
    return dict(original, mode="actual_early_moving_successor_refresh_v1",
                nonterminal_eviction="oldest_unprotected_of_command_with_more_than_two_slots",
                nonterminal_selection="continue_actual_path_then_alternate_commands_early_moving_root",
                moving_prestate_speed_mps=0.1,
                moving_anchor="earliest_tick_latest_arriving_episode_with_moving_rejection",
                moving_anchor_tie="latest_collection_id",
                stopped_only_episode_preserves_moving_anchor=True,
                first_preferred_command="fixed_keep",
                nonterminal_fallback="latest_rejected_source_of_selected_command",
                protected_script_sha256=original["script_sha256"],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def source_speed(snapshot, boundary, step):
    if step.result.terminated or step.result.truncated or step.result.check_result.valid:
        return None
    velocity = snapshot.planning_anchor(boundary.token).velocity
    if len(velocity) != 2 or not all(math.isfinite(float(v)) for v in velocity):
        raise ValueError("moving source requires finite actual XY anchor velocity")
    return math.hypot(*velocity)


class EarlyMovingBank(RejectionRefresh):
    def __init__(self):
        super().__init__()
        self.anchors = {}
        self.speeds = {}
        self.next_provider = PROVIDERS[0]

    def after_base(self, snapshot, boundary, step, **identity):
        provider = identity["provider"]
        if provider not in PROVIDERS:
            raise ValueError("early moving refresh requires fixed KEEP or FOLLOW")
        speed = source_speed(snapshot, boundary, step)
        super().after_base(snapshot, boundary, step, **identity)
        self.entries = type(self.entries)(e for e in self.entries if e.outcome is None)
        if speed is not None:
            current = self.entries[-1]
            self.speeds[current.collection_id] = speed
            old = next((e for e in self.entries
                        if e.collection_id == self.anchors.get(provider)), None)
            if speed >= 0.1 and (old is None or
                    old.boundary.token.episode_uid != boundary.token.episode_uid or
                    boundary.token.sim_tick <= old.boundary.token.sim_tick):
                self.anchors[provider] = current.collection_id
        if len(self.entries) > 4:
            counts = Counter(e.provider for e in self.entries)
            index = next(i for i, e in enumerate(self.entries)
                         if counts[e.provider] > 2 and
                         e.collection_id != self.anchors.get(e.provider))
            del self.entries[index]
            if index < self.cursor:
                self.cursor -= 1
        live = {e.collection_id for e in self.entries}
        self.source_checks = {i: c for i, c in self.source_checks.items() if i in live}
        self.speeds = {i: v for i, v in self.speeds.items() if i in live}
        assert set(self.anchors.values()) <= live

    def draw(self):
        if (not self.entries or not self.base_steps or self.base_steps % 4 or
                self.last_draw_base == self.base_steps):
            return None
        available = {e.provider for e in self.entries}
        other = PROVIDERS[1] if self.next_provider == PROVIDERS[0] else PROVIDERS[0]
        provider = self.next_provider if self.next_provider in available else other
        source_id = self.anchors.get(provider)
        selected = next((e for e in self.entries if e.collection_id == source_id), None)
        if selected is None:
            selected = next(e for e in reversed(self.entries) if e.provider == provider)
        self.cursor = next(i for i, e in enumerate(self.entries) if e is selected)
        result = super().draw()
        self.next_provider = PROVIDERS[1] if provider == PROVIDERS[0] else PROVIDERS[0]
        result[2].update(source_speed_mps=self.speeds[selected.collection_id],
                         root_selection="early_moving" if source_id is not None else "latest_fallback")
        return result


class EarlyMovingSuccessorRefresh(SuccessorRefresh):
    def __init__(self):
        super().__init__()
        self.banks["nonterminal"] = EarlyMovingBank()

    def after_base(self, snapshot, boundary, step, **identity):
        if identity["provider"] not in PROVIDERS:
            raise ValueError("early moving refresh requires fixed KEEP or FOLLOW")
        # Validate motion before either bank mutates. The saved anchor is a
        # current physical state, not a recorded future trajectory or teacher.
        source_speed(snapshot, boundary, step)
        return super().after_base(snapshot, boundary, step, **identity)
