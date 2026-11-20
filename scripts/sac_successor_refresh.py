"""Continue one actual nonterminal refresh path across alternating refresh slots.

State selection only: the caller executes the original current Gaussian and one
full SAC update per new transition. Existing collectors remain frozen.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from scripts.diagnose_sac_q_values import observation_fingerprint
from scripts.sac_stratified_refresh import StratifiedRefresh, refresh_spec as stratified_spec


def refresh_spec():
    original = stratified_spec()
    return dict(original, mode="actual_successor_refresh_v1", active_paths=1,
                retained_evicted_roots=1, max_retained_environments=10,
                selection="alternate_available_stages_terminal_round_robin_nonterminal_active_path",
                nonterminal_selection="continue_actual_path_until_terminal_then_next_fifo_root",
                terminal_selection="one_actual_step_from_terminal_source",
                active_path_survives_root_eviction=True,
                stratified_bank_sha256=original["script_sha256"],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


@dataclass
class ActivePath:
    environment: object
    boundary: object
    fingerprint: str
    lineage: dict
    root_environment: object
    root_boundary: object


class SuccessorRefresh(StratifiedRefresh):
    def __init__(self):
        super().__init__()
        self.active = None
        self.pending = None
        self.last_actual_collection_id = 0

    def after_base(self, snapshot, boundary, step, **identity):
        if self.pending is not None:
            raise RuntimeError("refresh outcome must be recorded before another base step")
        if identity["collection_id"] <= self.last_actual_collection_id:
            raise ValueError("collection IDs must increase across actual base and refresh steps")
        self.verify_preserved()
        super().after_base(snapshot, boundary, step, **identity)
        self.last_actual_collection_id = identity["collection_id"]

    def verify_preserved(self):
        super().verify_preserved()
        if self.active is not None:
            path = self.active
            if (path.root_environment.token != path.root_boundary.token or
                    observation_fingerprint(path.root_environment._boundary().sac_observation)
                    != path.lineage["source_observation_sha256"]):
                raise RuntimeError("active path's saved root was mutated")
            # During a pending step only the drawn environment is allowed to advance.
            if self.pending is None or self.pending[2]["source_stage"] != "nonterminal":
                if (path.environment.token != path.boundary.token or
                        observation_fingerprint(path.environment._boundary().sac_observation)
                        != path.fingerprint):
                    raise RuntimeError("waiting successor environment was mutated")

    def draw(self):
        if self.pending is not None:
            raise RuntimeError("refresh outcome must be recorded before drawing again")
        if (not self.base_steps or self.base_steps % 4 or self.last_draw_base == self.base_steps):
            return None
        self.verify_preserved()
        available = lambda stage: bool(self.banks[stage].entries) or (
            stage == "nonterminal" and self.active is not None)
        other = "nonterminal" if self.next_stage == "terminal" else "terminal"
        stage = self.next_stage if available(self.next_stage) else other
        if not available(stage):
            return None
        root = None
        if stage == "nonterminal" and self.active is not None:
            environment, boundary = self.active.environment, self.active.boundary
            lineage = dict(self.active.lineage)
        else:
            result = self.banks[stage].draw()
            assert result is not None
            environment, boundary, lineage = result
            root = next(e for e in self.banks[stage].entries
                        if e.collection_id == lineage["source_collection_id"])
            lineage.update(refresh_path_id=self.refresh_steps + 1, refresh_path_step=0,
                           previous_refresh_collection_id=None)
        self.refresh_steps += 1
        self.last_draw_base = self.base_steps
        self.next_stage = "nonterminal" if stage == "terminal" else "terminal"
        lineage.update(kind="successor_refresh", source_stage=stage,
                       base_steps=self.base_steps, refresh_index=self.refresh_steps,
                       bank_collection_ids=[e.collection_id for e in self.banks[stage].entries],
                       stage_bank_collection_ids={k: [e.collection_id for e in b.entries]
                                                  for k, b in self.banks.items()},
                       refresh_pre_observation_sha256=observation_fingerprint(boundary.sac_observation))
        self.pending = environment, boundary, dict(lineage), root
        return environment, boundary, lineage

    def after_refresh(self, step, *, collection_id):
        if self.pending is None:
            raise RuntimeError("no pending actual refresh")
        environment, before, lineage, root = self.pending
        terminal = step.result.terminated or step.result.truncated
        if (type(collection_id) is not int or collection_id <= self.last_actual_collection_id or
                step.transition is None or step.result.pre_token != before.token or
                step.boundary.token.sim_tick != before.token.sim_tick + 1 or
                step.boundary.token.episode_uid != before.token.episode_uid or
                step.boundary.token != environment.token or
                step.transition.bootstrap != (not terminal)):
            raise ValueError("refresh must record its own actual one-tick transition and bootstrap")
        fingerprint = observation_fingerprint(step.boundary.sac_observation)
        if observation_fingerprint(environment._boundary().sac_observation) != fingerprint:
            raise ValueError("refresh next observation differs from actual environment")
        self.verify_preserved()
        if lineage["source_stage"] == "nonterminal":
            if terminal:
                self.active = None
            else:
                root_environment = root.environment if root else self.active.root_environment
                root_boundary = root.boundary if root else self.active.root_boundary
                lineage.update(refresh_path_step=lineage["refresh_path_step"] + 1,
                               previous_refresh_collection_id=collection_id)
                self.active = ActivePath(environment, step.boundary, fingerprint, lineage,
                                         root_environment, root_boundary)
        self.pending = None
        self.last_actual_collection_id = collection_id
        self.verify_preserved()
