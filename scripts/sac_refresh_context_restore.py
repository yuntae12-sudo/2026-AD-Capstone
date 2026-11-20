"""Rebuild a fresh run's live collection context from its actual action log.

This is restoration, not experience collection: no policy forward pass, learner
update or replay insertion occurs. The currently supported cut is an unfinished
main episode after its optional refresh. Mid-branch cuts fail explicitly.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass
class RestoredContext:
    environment: object
    boundary: object
    failure_branches: object
    refresh_bank: object
    scene_rng: object
    scene_permutation: object
    scene_position: int
    policy_rng: object
    evidence: dict


def validate_cut(report, rows):
    """Reject boundaries the restoration protocol does not implement."""
    if (report.get("status") != "COMPLETE" or report.get("inherited_online_updates") != 0
            or report.get("continuation_history") or report.get("inherited_self_warmup") != 5000
            or report.get("updates") != len(rows) or report.get("new_transitions") != len(rows)
            or not rows):
        raise ValueError("requires a complete fresh own-experience run")
    if any(r["collection_id"] != i or r["update"] != i
           or r["policy_updates_before"] != i-1 or r["delta_ticks"] != 1
           for i, r in enumerate(rows, 1)):
        raise ValueError("actual one-call/one-update history is incomplete")
    base = [r for r in rows if r["lineage"]["kind"] != "successor_refresh"]
    if (not base or base[-1]["lineage"]["kind"] != "main" or not base[-1]["bootstrap"]
            or report["episodes"][-1]["outcome"] != "DIAGNOSTIC_BUDGET"):
        raise ValueError("only an unfinished main-episode cut is supported")
    collection = report["collection"]
    if (not collection.get("complete_path") or collection.get("samples") != 1
            or collection.get("window") != 6 or collection.get("selection") != "moving_predecessors"
            or collection.get("rollout_horizon") != 200):
        raise ValueError("requires the audited complete-path branch configuration")


def restore_context(directory, checkpoint, manifest, *, progress=None):
    """Return live simulator/collector objects after checking every stored step.

Episode UUIDs are restored before any action, so resumed lineage retains original
identity. Reset-created runtime objects are otherwise rebuilt by actual steps.
Caller owns subsequent sampling/training and must preserve all experiment inputs.
"""
    import jax
    import numpy as np
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from scripts.sac_failure_branches import FailureBranches
    from scripts.sac_failure_complete_path import choose_root
    from src.contracts.planning import to_wire
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.training.sac.collector import collect_step
    from src.training.sac.policy import proposal_from_action
    from src.training.sac.replay import Replay, pack_observation
    from src.training.sac.runtime import selected_scenes, versions

    directory, checkpoint, manifest = map(Path, (directory, checkpoint, manifest))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = json.loads((directory / "report.json").read_text())
    rows = [json.loads(x) for x in (directory / "ticks.jsonl").read_text().splitlines()]
    validate_cut(report, rows)
    if report["versions"] != versions(manifest):
        raise ValueError("runtime or data version changed")
    if "early_moving_refresh" in report:
        from scripts.audit_sac_early_moving_refresh import audit
        from scripts.sac_early_moving_refresh import EarlyMovingSuccessorRefresh as Bank
    elif "provider_protected_refresh" in report:
        from scripts.audit_sac_provider_protected_refresh import audit
        from scripts.sac_provider_protected_refresh import ProviderProtectedSuccessorRefresh as Bank
    else:
        raise ValueError("unsupported refresh contract")
    audited = audit(directory, checkpoint, manifest)
    arrays = Replay.load(directory / "replay.npz", report["versions"]).arrays
    inputs = {str(directory/n): sha(directory/n) for n in
              ("report.json", "ticks.jsonl", "updates.jsonl", "replay.npz", "online-learner.msgpack")}
    inputs[str(manifest)] = sha(manifest)
    inputs[str(Path(__file__))] = sha(__file__)
    candidates = selected_scenes(manifest, "train")
    scene_rng = np.random.default_rng(report["collection_seed"])
    order = scene_rng.permutation(len(candidates))
    position = -1
    bank = Bank()
    main_env = main_boundary = branches = branch_env = branch_boundary = None
    main_last = pending = branch_previous_id = None
    current_uid = None
    episode_pairs = []
    policy_rng = jax.random.PRNGKey(report["collection_seed"])

    def check_observation(boundary, i, prefix=""):
        for field, value in zip(("obs", "fields", "items"), pack_observation(boundary.sac_observation)):
            target = arrays[prefix+field][5000+i]
            if field == "obs":
                np.testing.assert_allclose(value, target, rtol=0, atol=1e-6)
            else:
                np.testing.assert_array_equal(value, target)

    for i, row in enumerate(rows):
        kind = row["lineage"]["kind"]
        if kind == "main":
            if row["episode_uid"] != current_uid:
                if main_last is not None and main_last.transition.bootstrap:
                    raise ValueError("main episode changes before real termination")
                position += 1
                if position == len(order):
                    order = scene_rng.permutation(len(candidates)); position = 0
                scene, path = candidates[order[position]]
                if (scene["scene_id"], scene["provider"]) != (row["scene_id"], row["provider"]):
                    raise ValueError("stored main order differs from seeded permutation")
                inputs[str(path)] = sha(path)
                main_env = SACPlanningEnvironment()
                main_env.reset(parse_scenario(path.read_bytes()), seed=report["seed"], provider_spec=row["provider"])
                # Reset has no action ledger; preserve the original identity before
                # rebuilding any cached proposal, PD or failure-history state.
                main_env.episode_uid = current_uid = row["episode_uid"]
                main_boundary = main_env._boundary()
                branches = FailureBranches(1, 6, "moving_predecessors")
                branch_env = branch_boundary = None
                episode_pairs.append((row["scene_id"], row["provider"]))
            branches.remember(main_env, main_boundary)
            environment, boundary = main_env, main_boundary
        elif kind == "failure_branch":
            lineage = row["lineage"]
            if lineage["branch_step"] == 0:
                if main_last is None or main_last.transition.bootstrap:
                    raise ValueError("branch has no actual terminal trigger")
                branch_env, branch_boundary, root = choose_root(list(branches.generate(main_last.result)))
                for key, value in root.items():
                    if lineage[key] != value:
                        raise ValueError(f"branch root mismatch: {key}")
                branch_previous_id = None
            if lineage["previous_collection_id"] != branch_previous_id:
                raise ValueError("branch predecessor mismatch")
            environment, boundary = branch_env, branch_boundary
            if observation_fingerprint(boundary.sac_observation) != lineage["observation_sha256"]:
                raise ValueError("branch observation differs")
        elif kind == "successor_refresh":
            if pending is None:
                raise ValueError("unexpected refresh without a scheduled draw")
            environment, boundary, lineage = pending
            if lineage != row["lineage"]:
                raise ValueError("reconstructed refresh selection/lineage differs")
        else:
            raise ValueError(f"unsupported collection kind: {kind}")
        if (kind == "successor_refresh") != (pending is not None):
            raise ValueError("missing or reordered scheduled refresh")
        if environment.token != boundary.token or to_wire(boundary.token) != row["raw_proposal"]["observation_token"]:
            raise ValueError("restored observation token differs")
        check_observation(boundary, i)
        snapshot = fork_environment(environment) if kind != "successor_refresh" else None
        actual = collect_step(environment, boundary,
            lambda e, token: proposal_from_action(e, token, arrays["raw_u"][5000+i]))
        if actual.transition is None:
            raise ValueError("restored step produced no actual transition")
        check_observation(actual.boundary, i, "next_")
        np.testing.assert_allclose(actual.reward.training, row["reward"], rtol=0, atol=1e-8)
        result = actual.result
        for key, value in dict(check_valid=result.check_result.valid,
                check_reason=result.check_result.reason, source=result.source.value,
                terminated=result.terminated, truncated=result.truncated,
                terminal_reason=result.terminal_reason.value if result.terminal_reason else None,
                bootstrap=actual.transition.bootstrap).items():
            if row[key] != value:
                raise ValueError(f"physical step {i+1} differs: {key}")
        for key, value in to_wire(result.actual_control).items():
            if key in ("acceleration_mps2", "curvature_per_m"):
                np.testing.assert_allclose(value, row["actual_control"][key], rtol=0, atol=1e-8)
            elif value != row["actual_control"][key]:
                raise ValueError(f"restored control differs: {key}")
        if kind == "successor_refresh":
            bank.after_refresh(actual, collection_id=i+1)
            pending = None
        else:
            bank.after_base(snapshot, boundary, actual, collection_id=i+1,
                scene_id=row["scene_id"], provider=row["provider"], kind=kind)
            if kind == "main":
                branches.after_step(result)
                main_boundary, main_last = actual.boundary, actual
            else:
                branch_boundary, branch_previous_id = actual.boundary, i+1
            if i+1 < len(rows):
                pending = bank.draw()
        policy_rng, _ = jax.random.split(policy_rng)
        if progress and ((i+1) % 100 == 0 or i+1 == len(rows)):
            progress(i+1, len(rows))
    bank.verify_preserved()
    if pending is not None or main_last is None or not main_last.transition.bootstrap:
        raise ValueError("restoration did not finish at an actionable main cut")
    if episode_pairs != [(e["scene_id"], e["provider"]) for e in report["episodes"]]:
        raise ValueError("episode inventory mismatch")
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"input changed during reconstruction: {path}")
    evidence = dict(status="COMPLETE", qualified=False, input_sha256=inputs,
        actual_replayed_steps=len(rows), actual_resets=len(episode_pairs),
        original_audit=audited, main_episode_uid=current_uid, main_tick=main_env.tick,
        main_observation_sha256=observation_fingerprint(main_boundary.sac_observation),
        bank_ids={s: [e.collection_id for e in b.entries] for s,b in bank.banks.items()},
        active_lineage=bank.active.lineage if bank.active else None,
        policy_rng=np.asarray(policy_rng).tolist(), scene_position=position,
        new_policy_calls=0, new_training_updates=0, replay_insertions=0,
        limits=["Live collector reconstruction only; no resumed SAC update or performance claim.",
                "Only fresh runs ending at an unfinished main episode are supported."])
    return RestoredContext(main_env, main_boundary, branches, bank, scene_rng,
        order, position, policy_rng, evidence)
