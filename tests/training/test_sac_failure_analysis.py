import copy
import json
from pathlib import Path
import numpy as np
import pytest

from scripts.sac_failure_analysis import (
    aggregate,
    analyze,
    evaluation_trend,
    freeze_trace,
    episode_groups,
    select_collisions,
    speed_bucket,
    summarize_episode,
)


def rows():
    result = []
    for tick, speed in ((1, 3.0), (2, 0.0)):
        result.append(
            dict(
                episode_uid="episode",
                scene_id="scene",
                provider="fixed_keep",
                tick=tick,
                raw_proposal=dict(
                    generated_tick=tick - 1, anchor_velocity_world=[speed, 0.0]
                ),
                sample_valid=True,
                check_valid=False,
                check_reason="PREDICTED_COLLISION",
                source="EMERGENCY",
                reward=-20.0 if tick == 2 else -0.01,
                actual_control=dict(acceleration_mps2=-2.0),
                event_flags=["COLLISION"] if tick == 2 else [],
                terminated=tick == 2,
                truncated=False,
                terminal_reason="COLLISION" if tick == 2 else None,
            )
        )
    return result


def metadata():
    return dict(
        versions={"v": 1},
        seed=0,
        training_mode="pure-sac",
        warmup_transitions=5000,
        updates=0,
        episodes=[
            dict(
                scene_id="scene",
                provider="fixed_keep",
                ticks=2,
                outcome="COLLISION",
                accepted=0,
                return_training=-20.01,
                stationary_ticks=2,
            )
        ],
    )


def test_live_snapshot_ignores_partial_tail_and_preserves_writer(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    out = tmp_path / "snapshot"
    out.mkdir()
    data = b"".join((json.dumps(r) + "\n").encode() for r in rows()) + b'{"unfinished":'
    (run / "ticks.jsonl").write_bytes(data)
    (run / "report.json").write_text(json.dumps(metadata()))
    meta, snapshot = freeze_trace(run, out)
    assert snapshot["lines"] == 2 and snapshot["incomplete_tail_bytes"] == len(
        b'{"unfinished":'
    )
    assert (run / "ticks.jsonl").read_bytes() == data
    report = analyze(out / "trace.snapshot.jsonl", meta)
    assert report["snapshot_phase"] == "WARMUP"
    assert report["summary"]["collision_rate"] == 1.0


def test_post_stationarity_never_substitutes_pre_state_and_reports_coverage():
    data = rows()
    meta = metadata()
    ep = summarize_episode(data, 0, meta, 0)
    assert ep["initial_speed"] == 3.0
    assert ep["stationary_ticks"] == 2 and ep["post_speed_observed_ticks"] == 2
    assert ep["terminal_rejection_streak"] == 2 and ep["terminal_braking_streak"] == 2
    assert ep["tail"][-1]["post_speed"] is None
    meta["episodes"] = []
    ep = summarize_episode(data, 0, meta, 0)
    assert ep["stationary_ticks"] == 1 and ep["post_speed_observed_ticks"] == 1
    assert aggregate([ep])["post_speed_coverage"] == 0.5
    meta = metadata()
    meta["episodes"][0]["stationary_ticks"] = 0
    with pytest.raises(ValueError, match="contradicts"):
        summarize_episode(data, 0, meta, 0)


def test_incomplete_and_corrupt_episodes_do_not_improve_outcome_denominators():
    ep = summarize_episode(rows(), 0, metadata(), 0)
    partial = copy.deepcopy(ep)
    partial.update(completed=False, outcome="IN_PROGRESS")
    corrupt = copy.deepcopy(ep)
    corrupt.update(integrity_valid=False, outcome="DATA_CORRUPTION")
    r = aggregate([ep, partial, corrupt])
    assert r["episodes"] == 1 and r["in_progress"] == 1 and r["corrupt_episodes"] == 1
    assert r["collision_rate"] == 1 and r["success_rate"] == 0
    assert aggregate([partial])["collision_rate"] is None


def test_phase_speed_boundaries_and_collision_selection_are_explicit():
    assert [speed_bucket(v) for v in (0, 0.1, 2, 10)] == [
        "stationary [0,0.1)",
        "low [0.1,2)",
        "medium [2,10)",
        "high [10,+inf)",
    ]
    ep = summarize_episode(rows(), 0, metadata(), 4999)
    assert ep["phase"] == "MIXED"
    ep2 = copy.deepcopy(ep)
    ep2.update(episode_uid="follow", behavior="FOLLOW")
    assert select_collisions([ep, copy.deepcopy(ep), ep2], 2) == [ep, ep2]
    assert select_collisions([ep], 0) == []


def test_discontinuous_trace_is_not_silently_accepted(tmp_path):
    data = rows()
    data[-1]["tick"] = 3
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in data))
    with pytest.raises(ValueError, match="noncontiguous"):
        list(episode_groups(path))


def test_reward_stationarity_warning_requires_paired_validation(tmp_path):
    from tests.training.test_sac_experiment import evaluation
    from scripts.sac_experiment_runtime import summarize_episodes

    paths = []
    for i in (1, 2):
        r = evaluation(f"PURE_SAC-{i*5000}", 0)
        for ep in r["episodes"]:
            ep["stationary_ticks"] = 3 + i
            ep["return_training"] = -20.0 + i
        r["by_behavior"] = summarize_episodes(r["episodes"])
        p = tmp_path / f"{i}.json"
        p.write_text(json.dumps(r))
        paths.append(p)
    assert (
        evaluation_trend(paths[:1], {"v": 1}, 0)["status"]
        == "INSUFFICIENT_PAIRED_EVALUATIONS"
    )
    trend = evaluation_trend(paths, {"v": 1}, 0)
    assert len(trend["comparisons"]) == 2
    assert all(r["reward_up_and_stationary_up"] for r in trend["comparisons"])
    r = json.loads(paths[-1].read_text())
    r["episodes"][0]["scene_id"] = "other"
    paths[-1].write_text(json.dumps(r))
    with pytest.raises(ValueError, match="unpaired"):
        evaluation_trend(paths, {"v": 1}, 0)


def test_saved_action_replay_verifies_controls_before_contact_attribution():
    from tests.environment.test_womd_control import scene
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.training.sac.collector import collect_step
    from src.training.sac.teacher import FrenetTeacher
    from src.contracts.planning import to_wire
    from scripts.sac_failure_replay import replay_episode

    scenario = scene()
    e = SACPlanningEnvironment()
    b = e.reset(scenario, provider_spec="fixed_keep", seed=0)
    teacher = FrenetTeacher()
    trace = []
    for _ in range(2):
        step = collect_step(e, b, lambda env, t: teacher.propose(env, t).proposal)
        r, b = step.result, step.boundary
        trace.append(
            dict(
                episode_uid=e.episode_uid,
                scene_id=scenario.scenario_id,
                provider="fixed_keep",
                tick=e.tick,
                raw_proposal=to_wire(r.raw_proposal),
                source=r.source.value,
                check_valid=r.check_result.valid,
                check_reason=r.check_result.reason,
                actual_control=to_wire(r.actual_control),
                event_flags=list(r.event_flags),
                terminated=r.terminated,
                truncated=r.truncated,
                sample_valid=r.sample_valid,
                terminal_reason=r.terminal_reason.value if r.terminal_reason else None,
                reward=step.reward.training,
            )
        )
    result = replay_episode(trace, scenario, 0)
    assert result["status"] == "VERIFIED" and result["verified_post_anchors"] == 1
    trace[-1]["actual_control"]["acceleration_mps2"] += 1
    with pytest.raises(ValueError, match="divergence"):
        replay_episode(trace, scenario, 0)
