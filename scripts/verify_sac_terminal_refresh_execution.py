#!/usr/bin/env python3
"""Replay predeclared first refresh per provider/source kind using actual Waymax steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "checkpoint", "manifest", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    from scripts.audit_sac_terminal_refresh import audit
    from scripts.sac_branch_reconstruction import branch_history_indices
    from scripts.diagnose_sac_q_values import fork_environment, observation_fingerprint
    from src.training.sac.runtime import selected_scenes, versions
    from src.training.sac.collector import collect_step
    from src.training.sac.policy import proposal_from_action
    from src.training.sac.replay import Replay, pack_observation
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from src.contracts.planning import to_wire

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audited = audit(args.directory, args.checkpoint, args.manifest)
    report = json.loads((args.directory / "report.json").read_text())
    assert report["versions"] == versions(args.manifest)
    arrays = Replay.load(args.directory / "replay.npz", report["versions"]).arrays
    rows = [json.loads(x) for x in (args.directory / "ticks.jsonl").read_text().splitlines()]
    selected = {}
    for i, row in enumerate(rows):
        if row["lineage"]["kind"] == "terminal_refresh":
            original = rows[row["lineage"]["source_collection_id"]-1]
            selected.setdefault((row["provider"], original["lineage"]["kind"]), i)
    assert selected, "no actual refresh to replay"
    paths = {(r["scene_id"], r["provider"]): p for r, p in selected_scenes(args.manifest, "train")}
    inputs = {str(args.directory/name): sha(args.directory/name)
              for name in ("report.json", "online-learner.msgpack", "replay.npz", "ticks.jsonl", "updates.jsonl")}

    def check_obs(boundary, i, prefix=""):
        for field, actual in zip(("obs", "fields", "items"), pack_observation(boundary.sac_observation)):
            expected = arrays[prefix+field][5000+i]
            if field == "obs":
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            else:
                np.testing.assert_array_equal(actual, expected)

    def execute(env, boundary, i):
        row = rows[i]
        check_obs(boundary, i)
        assert env.tick == row["pre_tick"]
        actual = collect_step(env, boundary,
            lambda e, t: proposal_from_action(e, t, arrays["raw_u"][5000+i]))
        assert actual.transition is not None
        np.testing.assert_allclose(actual.reward.training, row["reward"], rtol=0, atol=1e-8)
        assert actual.result.check_result.valid == row["check_valid"]
        assert actual.result.check_result.reason == row["check_reason"]
        assert actual.result.source.value == row["source"]
        assert actual.transition.bootstrap == row["bootstrap"]
        assert actual.result.terminated == row["terminated"] and actual.result.truncated == row["truncated"]
        reason = actual.result.terminal_reason.value if actual.result.terminal_reason else None
        assert reason == row["terminal_reason"]
        control = to_wire(actual.result.actual_control)
        assert control.keys() == row["actual_control"].keys()
        for key, value in control.items():
            if key in ("acceleration_mps2", "curvature_per_m"):
                np.testing.assert_allclose(value, row["actual_control"][key], rtol=0, atol=1e-8)
            else:
                assert value == row["actual_control"][key]
        check_obs(actual.boundary, i, "next_")
        return actual

    results = []
    for (provider, source_kind), i in selected.items():
        row = rows[i]; source_i = row["lineage"]["source_collection_id"]-1
        source = rows[source_i]
        if source_kind == "failure_branch":
            history = branch_history_indices(rows, source_i)
        else:
            history = [j for j, r in enumerate(rows[:source_i+1])
                       if r["lineage"]["kind"] == "main" and r["episode_uid"] == source["episode_uid"]]
        assert history[-1] == source_i
        assert [rows[j]["pre_tick"] for j in history] == list(range(source["pre_tick"]+1))
        env = SACPlanningEnvironment()
        path = paths[(source["scene_id"], provider)]; inputs[str(path)] = sha(path)
        boundary = env.reset(parse_scenario(path.read_bytes()), seed=report["seed"], provider_spec=provider)
        for j in history[:-1]:
            actual = execute(env, boundary, j)
            assert actual.transition.bootstrap
            boundary = actual.boundary
        before = observation_fingerprint(boundary.sac_observation)
        assert before == row["lineage"]["source_observation_sha256"]
        original_step = execute(fork_environment(env), boundary, source_i)
        refresh_step = execute(fork_environment(env), boundary, i)
        assert observation_fingerprint(env._boundary().sac_observation) == before
        results.append(dict(provider=provider, source_kind=source_kind,
            source_collection_id=source_i+1, refresh_collection_id=i+1,
            scene_id=row["scene_id"], pre_tick=row["pre_tick"], prefix_steps=len(history)-1,
            original_outcome=source["terminal_reason"], refresh_outcome=row["terminal_reason"],
            refresh_bootstrap=refresh_step.transition.bootstrap,
            original_reward=original_step.reward.training, refresh_reward=refresh_step.reward.training))
    for path, digest in inputs.items():
        assert sha(Path(path)) == digest
    result = dict(status="COMPLETE", qualified=False, cases=results,
                  source_audit=audited, input_sha256=inputs, script_sha256=sha(Path(__file__)),
                  note="First actual refresh per provider/source kind; replay prefix and both actual actions through original collector. No learner/replay writes or independent performance claim.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
