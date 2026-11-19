"""Diagnostic-only real simulator branches from recent train failure states.

Snapshots preserve the parent episode/token. Branch lineage is separate metadata,
not an invented reset or success. Each yielded environment must be stepped through
the original collector; no reward, terminal flag or next observation is fabricated.
"""

from collections import deque

from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint


class FailureBranches:
    def __init__(self, samples=0, window=6):
        if not isinstance(samples, int) or not 0 <= samples <= 64:
            raise ValueError("branch samples must be 0..64")
        if not isinstance(window, int) or not 1 <= window <= 32:
            raise ValueError("branch window must be 1..32")
        self.samples = samples
        self.snapshots = deque(maxlen=window)

    def remember(self, environment, boundary):
        if self.samples:
            if environment.token != boundary.token or not boundary.needs_sac_action:
                raise ValueError("branch snapshot requires current policy boundary")
            self.snapshots.append((fork_environment(environment), boundary))

    def generate(self, result):
        if not (result.terminated or result.truncated):
            return
        if result.terminal_reason.value not in ("COLLISION", "OFFROAD"):
            return
        # Round-robin newest to oldest: a budget cutoff does not spend all
        # remaining experience on just one state. No best-action selection.
        for sample in range(self.samples):
            for snapshot, boundary in reversed(self.snapshots):
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
                    observation_sha256=observation_fingerprint(boundary.sac_observation),
                )
                if snapshot.token != boundary.token or observation_fingerprint(snapshot._boundary().sac_observation) != observation_fingerprint(boundary.sac_observation):
                    raise RuntimeError("branch mutated its source snapshot")
