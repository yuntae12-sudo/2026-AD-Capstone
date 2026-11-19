"""Resolve a saved branch's actual history, without substituting main actions."""


def branch_history_indices(rows, index):
    """Return main prefix then branch prefix, including the target pre-state row."""
    target = rows[index]
    root = target["lineage"]
    if root["kind"] != "failure_branch":
        raise ValueError("requires a saved failure branch")
    identity = (root["parent_episode_uid"], root["parent_tick"], root["branch_sample"])
    chain = [index]
    while rows[chain[-1]]["lineage"].get("branch_step", 0):
        current = rows[chain[-1]]
        lineage = current["lineage"]
        previous_id = lineage.get("previous_collection_id")
        if not isinstance(previous_id, int) or not 1 <= previous_id <= chain[-1]:
            raise ValueError("invalid previous actual branch transition")
        previous_index = previous_id - 1
        previous = rows[previous_index]
        pl = previous["lineage"]
        if (pl["kind"] != "failure_branch"
                or (pl["parent_episode_uid"], pl["parent_tick"], pl["branch_sample"]) != identity
                or pl.get("branch_step", 0) + 1 != lineage["branch_step"]
                or previous["pre_tick"] + 1 != current["pre_tick"]
                or not previous["bootstrap"]
                or previous["scene_id"] != current["scene_id"]
                or previous["provider"] != current["provider"]):
            raise ValueError("broken actual branch continuation")
        chain.append(previous_index)
    chain.reverse()
    first = rows[chain[0]]
    if first["pre_tick"] != identity[1] or first["lineage"].get("previous_collection_id") is not None:
        raise ValueError("branch root does not match parent tick")
    main = [i for i, row in enumerate(rows[:chain[0]])
            if row["lineage"]["kind"] == "main"
            and row["episode_uid"] == identity[0]
            and row["pre_tick"] < identity[1]]
    if [rows[i]["pre_tick"] for i in main] != list(range(identity[1])):
        raise ValueError("incomplete actual main prefix")
    for i in main:
        row = rows[i]
        if not row["bootstrap"] or row["scene_id"] != target["scene_id"] or row["provider"] != target["provider"]:
            raise ValueError("invalid main prefix before branch root")
    return main + chain
