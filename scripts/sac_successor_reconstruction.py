"""Resolve a refresh root's real history and its actual interleaved successor path."""
from scripts.sac_branch_reconstruction import branch_history_indices


def successor_history_indices(rows, index):
    target = rows[index]
    lineage = target["lineage"]
    if lineage["kind"] != "successor_refresh":
        raise ValueError("requires a successor refresh")
    source_id = lineage["source_collection_id"]
    path_id = lineage["refresh_path_id"]
    if type(source_id) is not int or not 1 <= source_id <= index:
        raise ValueError("invalid source ID")
    source = rows[source_id-1]
    chain = [index]
    while rows[chain[-1]]["lineage"]["refresh_path_step"]:
        current = rows[chain[-1]]
        cl = current["lineage"]
        previous_id = cl["previous_refresh_collection_id"]
        if type(previous_id) is not int or not 1 <= previous_id <= chain[-1]:
            raise ValueError("invalid predecessor ID")
        previous = rows[previous_id-1]
        pl = previous["lineage"]
        if (pl["kind"] != "successor_refresh" or pl["source_stage"] != "nonterminal" or
                pl["source_collection_id"] != source_id or pl["refresh_path_id"] != path_id or
                pl["refresh_path_step"]+1 != cl["refresh_path_step"] or
                not previous["bootstrap"] or previous["pre_tick"]+1 != current["pre_tick"] or
                any(previous[k] != target[k] for k in ("scene_id", "provider", "episode_uid"))):
            raise ValueError("broken actual successor path")
        chain.append(previous_id-1)
    chain.reverse()
    first = rows[chain[0]]
    if any(any(rows[i][k] != source[k] for k in ("scene_id", "provider", "episode_uid")) for i in chain):
        raise ValueError("successor changed source scene, provider, or episode")
    if lineage["refresh_path_step"] and lineage["source_stage"] != "nonterminal":
        raise ValueError("only nonterminal sources continue")
    if (first["pre_tick"] != source["pre_tick"] or
            first["lineage"]["previous_refresh_collection_id"] is not None or
            first["lineage"]["source_collection_id"] != source_id or
            first["lineage"]["refresh_path_id"] != path_id):
        raise ValueError("refresh root differs from actual source")
    if source["lineage"]["kind"] == "failure_branch":
        history = branch_history_indices(rows, source_id-1)[:-1]
    elif source["lineage"]["kind"] == "main":
        history = [i for i, row in enumerate(rows[:source_id-1])
                   if row["lineage"]["kind"] == "main" and row["episode_uid"] == source["episode_uid"]]
    else:
        raise ValueError("recursive source prohibited")
    result = history + chain
    if [rows[i]["pre_tick"] for i in result] != list(range(target["pre_tick"]+1)):
        raise ValueError("incomplete actual history")
    if any(not rows[i]["bootstrap"] for i in result[:-1]):
        raise ValueError("history crosses actual termination")
    return result
