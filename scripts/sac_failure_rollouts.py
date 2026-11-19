"""Bounded real continuations from train failure roots; no synthetic terminals.

This opt-in collection experiment changes only visitation. The callback must use
the original stochastic policy, collector, replay and one update per actual step.
One path per selected root is advanced round-robin. A horizon or global budget
cutoff keeps the last actual transition's bootstrap/termination untouched.
"""

from scripts.diagnose_sac_q_values import observation_fingerprint


def collect_failure_rollouts(branches, result, *, horizon, train_step, can_step):
    """Run actual paths, returning their count; callback returns (step, collection_id).

    ``branches.samples`` must be one: repeated independent root actions and a
    continuous suffix are separate experiments. Lineage identifies each root,
    the previous actual transition and the current observation independently.
    """
    if not isinstance(horizon, int) or not 2 <= horizon <= 8:
        raise ValueError("failure rollout horizon must be 2..8")
    if branches.samples != 1 or branches.window * horizon > 48:
        raise ValueError("rollouts require one path per root and at most48 steps per trigger")
    active = []
    for environment, boundary, lineage in branches.generate(result):
        active.append((environment, boundary, lineage, None))
    count = 0
    for depth in range(horizon):
        following = []
        for environment, boundary, root, previous_id in active:
            if not can_step():
                return count
            if environment.token != boundary.token or not boundary.needs_sac_action:
                raise RuntimeError("rollout boundary is not current and actionable")
            lineage = dict(root, branch_step=depth,
                           previous_collection_id=previous_id,
                           root_observation_sha256=root["observation_sha256"],
                           observation_sha256=observation_fingerprint(boundary.sac_observation))
            step, collection_id = train_step(environment, boundary, lineage)
            if (step.transition is None or step.result.pre_token != boundary.token
                    or step.boundary.token != environment.token
                    or environment.token.sim_tick != boundary.token.sim_tick + 1):
                raise RuntimeError("callback must execute exactly one actual step")
            if not isinstance(collection_id, int) or collection_id <= (previous_id or 0):
                raise RuntimeError("invalid actual collection sequence")
            if step.transition.bootstrap != (not (step.result.terminated or step.result.truncated)):
                raise RuntimeError("callback changed actual termination semantics")
            count += 1
            if not (step.result.terminated or step.result.truncated):
                following.append((environment, step.boundary, root, collection_id))
        active = following
        if not active:
            break
    return count
