"""Preserve each root's actual quota, then finish one still-open failure path.

The quota prefix uses the existing collector unchanged. The chosen root depends
only on past main outcomes. No new action is filtered, and an already terminal
last quota step is never restarted merely to obtain an extension.
"""

from scripts.diagnose_sac_q_values import observation_fingerprint
from scripts.sac_failure_complete_path import choose_root
from scripts.sac_failure_rollout_budget import collect_fixed_budget_rollouts


def collect_quota_then_extension(branches, result, *, horizon, train_step, can_step):
    if (horizon != 8 or branches.samples != 8 or branches.window != 6
            or branches.selection != "moving_predecessors"):
        raise ValueError("quota extension requires samples8/window6/moving_predecessors/horizon8")
    roots = {}

    def remember_step(environment, boundary, lineage):
        key = lineage["parent_episode_uid"], lineage["parent_tick"]
        if key not in roots:
            roots[key] = dict(initial_boundary=boundary, initial_lineage=dict(lineage))
        step, collection_id = train_step(environment, boundary, lineage)
        roots[key].update(environment=environment, step=step,
                          collection_id=collection_id, lineage=dict(lineage))
        return step, collection_id

    count = collect_fixed_budget_rollouts(
        branches, result, horizon=horizon, train_step=remember_step, can_step=can_step)
    if not roots or not can_step():
        return count
    if any(r["lineage"]["root_budget_step"] != horizon - 1 for r in roots.values()):
        raise RuntimeError("extension requires every selected root's completed quota")
    selected = choose_root([
        (key, r["initial_boundary"], r["initial_lineage"]) for key, r in roots.items()])[0]
    root = roots[selected]
    step = root["step"]
    if step.result.terminated or step.result.truncated:
        return count
    environment, boundary = root["environment"], step.boundary
    previous_id, previous_lineage = root["collection_id"], root["lineage"]
    fingerprint = observation_fingerprint(root["initial_boundary"].sac_observation)
    for depth in range(previous_lineage["branch_step"] + 1, 200):
        if not can_step():
            return count
        if environment.token != boundary.token or not boundary.needs_sac_action:
            raise RuntimeError("quota extension boundary is not current and actionable")
        lineage = dict(previous_lineage, branch_step=depth,
                       root_budget_step=previous_lineage["root_budget_step"] + 1,
                       previous_collection_id=previous_id, quota_extension=True,
                       root_selection_rule="earliest_predecessor_then_rejection_then_recent",
                       observation_sha256=observation_fingerprint(boundary.sac_observation))
        step, collection_id = train_step(environment, boundary, lineage)
        if (step.transition is None or step.result.pre_token != boundary.token
                or step.boundary.token != environment.token
                or environment.token.sim_tick != boundary.token.sim_tick + 1):
            raise RuntimeError("callback must execute exactly one actual step")
        if not isinstance(collection_id, int) or collection_id <= previous_id:
            raise RuntimeError("invalid actual collection sequence")
        terminal = step.result.terminated or step.result.truncated
        if step.transition.bootstrap != (not terminal):
            raise RuntimeError("callback changed actual termination semantics")
        if observation_fingerprint(root["initial_boundary"].sac_observation) != fingerprint:
            raise RuntimeError("extension mutated its original root observation")
        count += 1
        if terminal:
            return count
        boundary, previous_id, previous_lineage = step.boundary, collection_id, lineage
    raise RuntimeError("original environment did not reach terminal within200 actual path steps")
