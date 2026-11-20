"""Bounded continuation metadata for a completed fixed-policy diagnosis.

This does not turn a budget boundary into a terminal transition and never
accepts an ordinary SAC training checkpoint as a frozen-policy parent.
"""

from copy import deepcopy


def continuation_plan(parent, last_tick, replay_metadata, total_steps):
    if type(total_steps) is not int or total_steps not in (1100, 2000, 5000):
        raise ValueError("bounded diagnostic totals are1100,2000,5000")
    if (parent.get("status") != "COMPLETE"
            or parent.get("scope") != "FIXED_POLICY_CRITIC_EVALUATION_NOT_SAC_TRAINING"
            or parent.get("qualified") is not False
            or parent.get("production_checkpoint") is not False):
        raise ValueError("requires completed unqualified fixed-policy diagnosis")
    if (parent.get("budget") != 1000 or parent.get("critic_steps") != 1000
            or parent.get("actual_transitions") != 1000
            or parent.get("inherited_replay_size") != 6000
            or parent.get("final_replay_size") != 7000
            or parent.get("actor_updates") != 0 or parent.get("temperature_updates") != 0
            or parent.get("actor_and_temperature_bytes_unchanged") is not True
            or parent.get("original_inputs_unchanged") is not True):
        raise ValueError("parent counters or frozen-policy evidence disagree")
    if (replay_metadata.get("size") != 7000 or replay_metadata.get("cursor") != 7000
            or replay_metadata.get("versions") != parent["versions"]
            or replay_metadata.get("rng") != parent["final_replay_rng"]
            or 6000 + total_steps >= replay_metadata["capacity"]):
        raise ValueError("replay continuation must preserve versions/RNG and avoid wrap")
    episodes = parent["episodes"]
    if not episodes or [ep["rollout"] for ep in episodes] != list(range(len(episodes))):
        raise ValueError("noncontiguous parent rollouts")
    last = episodes[-1]
    if (last_tick["collection_id"] != 1000 or last_tick["rollout"] != last["rollout"]
            or last_tick["branch_step"] + 1 != last["ticks"]
            or bool(last_tick["bootstrap"]) != (not (last_tick["terminated"] or last_tick["truncated"]))):
        raise ValueError("parent last transition is inconsistent")
    if last_tick["bootstrap"]:
        if last["outcome"] != "DIAGNOSTIC_BUDGET" or last_tick["outcome"] is not None:
            raise ValueError("budget boundary must retain bootstrap without fake terminal")
    elif last["outcome"] != last_tick["outcome"] or last["outcome"] == "DIAGNOSTIC_BUDGET":
        raise ValueError("real terminal must keep its actual outcome")
    for ep in episodes:
        if ep["policy_seed"] != parent["collection_seed"] + ep["rollout"] or ep["root_index"] != ep["rollout"] % 2:
            raise ValueError("parent rollout seed/root lineage mismatch")
    return dict(inherited_critic_steps=1000, total_critic_steps=total_steps,
                new_steps=total_steps-1000, original_replay_size=6000,
                inherited_replay_size=7000, final_replay_size=6000+total_steps,
                first_new_rollout=len(episodes), collection_seed=parent["collection_seed"],
                inherited_replay_rng=deepcopy(replay_metadata["rng"]),
                inherited_boundary=dict(after_collection_id=1000, rollout=last["rollout"],
                                        bootstrap=last_tick["bootstrap"], outcome=last["outcome"]),
                exact_simulator_resume=False)
