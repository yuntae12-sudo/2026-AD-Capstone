"""Continue actual failure branches with the same step quota for every root.

A terminal path is finished. If its root still has quota, start a new actual
path from the unchanged root snapshot. This preserves the original independent
sampler's number of actual steps per root, including immediate-terminal roots.
"""

from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint


def collect_fixed_budget_rollouts(branches, result, *, horizon, train_step, can_step):
    if (not isinstance(horizon, int) or not 2 <= horizon <= 8
            or branches.samples != horizon or branches.window * horizon > 48):
        raise ValueError("fixed root quota requires samples=horizon2..8 and at most48 steps")
    roots = []
    for environment, boundary, lineage in branches.generate(result):
        if lineage["branch_sample"] != 0:
            break
        roots.append(dict(environment=environment, boundary=boundary,
                          template=fork_environment(environment), root_boundary=boundary,
                          metadata=lineage, path=0, depth=0, previous_id=None))
    count = 0
    for budget_step in range(horizon):
        for root in roots:
            if not can_step():
                return count
            environment, boundary = root["environment"], root["boundary"]
            if environment.token != boundary.token or not boundary.needs_sac_action:
                raise RuntimeError("fixed-quota branch is not current and actionable")
            metadata = root["metadata"]
            lineage = dict(metadata, branch_sample=root["path"], branch_step=root["depth"],
                           root_budget_step=budget_step, previous_collection_id=root["previous_id"],
                           root_observation_sha256=metadata["observation_sha256"],
                           observation_sha256=observation_fingerprint(boundary.sac_observation))
            step, collection_id = train_step(environment, boundary, lineage)
            if (step.transition is None or step.result.pre_token != boundary.token
                    or step.boundary.token != environment.token
                    or environment.token.sim_tick != boundary.token.sim_tick + 1):
                raise RuntimeError("callback must execute exactly one actual step")
            if not isinstance(collection_id, int) or collection_id <= (root["previous_id"] or 0):
                raise RuntimeError("invalid actual collection sequence")
            terminal = step.result.terminated or step.result.truncated
            if step.transition.bootstrap != (not terminal):
                raise RuntimeError("callback changed actual termination semantics")
            count += 1
            if terminal:
                # A new path, not a continuation after a terminal transition.
                root.update(environment=fork_environment(root["template"]),
                            boundary=root["root_boundary"], path=root["path"] + 1,
                            depth=0, previous_id=None)
            else:
                root.update(boundary=step.boundary, depth=root["depth"] + 1,
                            previous_id=collection_id)
            if (root["template"].token != root["root_boundary"].token
                    or observation_fingerprint(root["template"]._boundary().sac_observation)
                    != metadata["observation_sha256"]):
                raise RuntimeError("actual rollout mutated its root snapshot")
    return count
