"""One real failure-predecessor path through the original terminal boundary.

Selection uses past main-trajectory outcomes only. Every new stochastic action,
including rejection/braking, goes through the unchanged collector and one SAC
update. A global collection budget may stop the path without changing bootstrap.
"""

from scripts.diagnose_sac_q_values import observation_fingerprint


def choose_root(candidates):
    """Prefer the earliest accepted predecessor, then moving rejection/recent."""
    priority = {"moving_predecessor": 0, "moving_rejection": 1,
                "recent": 2, "terminal": 3}
    return min(candidates, key=lambda c: (
        priority[c[2]["selection_role"]], c[1].token.sim_tick))


def collect_complete_failure_path(branches, result, *, horizon, train_step, can_step):
    if (horizon != 200 or branches.samples != 1 or branches.window != 6
            or branches.selection != "moving_predecessors"):
        raise ValueError("complete path requires horizon200, samples1, window6 and moving_predecessors")
    roots = list(branches.generate(result))
    if not roots or not can_step():
        return 0
    environment, boundary, root = choose_root(roots)
    original_boundary = boundary
    original_fingerprint = observation_fingerprint(boundary.sac_observation)
    previous_id = None
    for depth in range(horizon):
        if not can_step():
            return depth
        if environment.token != boundary.token or not boundary.needs_sac_action:
            raise RuntimeError("complete-path boundary is not current and actionable")
        lineage = dict(root, branch_step=depth, previous_collection_id=previous_id,
                       root_selection_rule="earliest_predecessor_then_rejection_then_recent",
                       root_observation_sha256=original_fingerprint,
                       observation_sha256=observation_fingerprint(boundary.sac_observation))
        step, collection_id = train_step(environment, boundary, lineage)
        if (step.transition is None or step.result.pre_token != boundary.token
                or step.boundary.token != environment.token
                or environment.token.sim_tick != boundary.token.sim_tick + 1):
            raise RuntimeError("callback must execute exactly one actual step")
        if not isinstance(collection_id, int) or collection_id <= (previous_id or 0):
            raise RuntimeError("invalid actual collection sequence")
        terminal = step.result.terminated or step.result.truncated
        if step.transition.bootstrap != (not terminal):
            raise RuntimeError("callback changed actual termination semantics")
        if observation_fingerprint(original_boundary.sac_observation) != original_fingerprint:
            raise RuntimeError("actual rollout mutated its root observation")
        if terminal:
            return depth + 1
        boundary, previous_id = step.boundary, collection_id
    raise RuntimeError("original environment did not reach terminal within200 actual steps")
