"""Diagnostic-only real simulator branches from recent train failure states.

Snapshots preserve the parent episode/token. Branch lineage is separate metadata,
not an invented reset or success. Each yielded environment must be stepped through
the original collector; no reward, terminal flag or next observation is fabricated.
"""

from collections import deque

from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint


class FailureBranches:
    def __init__(self, samples=0, window=6, selection="recent"):
        if not isinstance(samples, int) or not 0 <= samples <= 64:
            raise ValueError("branch samples must be 0..64")
        if not isinstance(window, int) or not 1 <= window <= 32:
            raise ValueError("branch window must be 1..32")
        if selection not in ("recent", "moving_rejections"):
            raise ValueError("unknown failure state selection")
        self.samples = samples
        self.window = window
        self.selection = selection
        self.snapshots = deque(maxlen=window)
        self.moving_rejections = []

    def remember(self, environment, boundary):
        if self.samples:
            if environment.token != boundary.token or not boundary.needs_sac_action:
                raise ValueError("branch snapshot requires current policy boundary")
            self.snapshots.append((fork_environment(environment), boundary))

    def after_step(self, result):
        """Remember early moving rejections, without changing the main rollout."""
        if not self.samples or self.selection != "moving_rejections":
            return
        snapshot, boundary = self.snapshots[-1]
        if result.pre_token != boundary.token:
            raise ValueError("failure selection must follow the captured main step")
        if (not result.check_result.valid and snapshot.ego.speed >= 0.1
                and len(self.moving_rejections) < self.window - 1):
            self.moving_rejections.append((snapshot, boundary))

    def selected_snapshots(self):
        if self.selection == "recent":
            return list(self.snapshots)
        # Keep the actual final pre-terminal state; replace other late snapshots
        # with the earliest moving rejections. Fill unused slots with recent
        # states so sample count stays matched when the episode is long enough.
        chosen = list(self.moving_rejections)
        last = self.snapshots[-1]
        if not any(b.token == last[1].token for _, b in chosen):
            chosen.append(last)
        for item in reversed(self.snapshots):
            if len(chosen) >= self.window:
                break
            if not any(b.token == item[1].token for _, b in chosen):
                chosen.append(item)
        return sorted(chosen, key=lambda item: item[1].token.sim_tick)

    def generate(self, result):
        if not (result.terminated or result.truncated):
            return
        if result.terminal_reason.value not in ("COLLISION", "OFFROAD"):
            return
        # Round-robin newest to oldest: a budget cutoff does not spend all
        # remaining experience on just one state. No best-action selection.
        selected = self.selected_snapshots() if self.snapshots else []
        for sample in range(self.samples):
            for snapshot, boundary in reversed(selected):
                branch = fork_environment(snapshot)
                if observation_fingerprint(branch._boundary().sac_observation) != observation_fingerprint(boundary.sac_observation):
                    raise RuntimeError("snapshot observation changed before execution")
                yield branch, boundary, dict(
                    kind="failure_branch",
                    parent_episode_uid=boundary.token.episode_uid,
                    parent_tick=boundary.token.sim_tick,
                    trigger_tick=result.pre_token.sim_tick,
                    trigger_outcome=result.terminal_reason.value,
                    branch_sample=sample,
                    state_selection=self.selection,
                    selection_role=("moving_rejection" if any(b.token == boundary.token for _, b in self.moving_rejections)
                                    else "terminal" if boundary.token.sim_tick == result.pre_token.sim_tick else "recent"),
                    observation_sha256=observation_fingerprint(boundary.sac_observation),
                )
                if snapshot.token != boundary.token or observation_fingerprint(snapshot._boundary().sac_observation) != observation_fingerprint(boundary.sac_observation):
                    raise RuntimeError("branch mutated its source snapshot")
