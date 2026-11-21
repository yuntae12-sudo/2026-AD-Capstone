import json
from pathlib import Path
import sys
import os
import signal
import subprocess
import time
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen, Request

import pytest

from scripts import sac_dashboard_data as data
from scripts import sac_experiment as runner
from scripts.serve_sac_dashboard import Handler


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_missing_evaluation_is_not_zero_success():
    assert data.summarize([])["success_rate"] is None
    assert data.summarize([])["collision_rate"] is None


def test_rates_use_tick_denominators_and_episode_reward():
    rows = [dict(scene_id="same", ticks=10, rejected=5, braking=4, stationary=2,
                 outcome="COLLISION", return_training=-20),
            dict(scene_id="same", ticks=30, rejected=3, braking=0, stationary=0,
                 outcome="ROUTE_GOAL", return_training=10)]
    s = data.summarize(rows)
    assert s["scenes"] == 1 and s["episodes"] == 2
    assert s["rejected"] == .2 and s["braking"] == .1
    assert s["success_rate"] == .5 and s["mean_return"] == -5


def test_curve_handles_partial_writes_and_nonfinite(tmp_path):
    path = tmp_path / "updates.jsonl"
    path.write_text('{"update":1,"alpha":0.1}\n{"update":2,"alpha":NaN}\n{"update":3')
    c = data.curve(path, limit=1)
    assert c["rows"] == 2 and c["points"] == [dict(update=2, alpha=.1, actor_loss=None,
        q1_loss=None, q2_loss=None, actor_q1_abs_max=None, target_abs_max=None)]


def test_inside_rejects_symlink_escape(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        data.inside(root / "escape/private.json", root)


def test_snapshot_displays_current_and_reused_separately(tmp_path):
    root = tmp_path / "run"
    model = root / "candidate120"
    write(root / "decision.json", dict(scope="MANUAL_REFRESH_CONTINUATION", seed=0,
          active_arms=["candidate"], total_updates_per_arm=120, inherited_updates_per_arm=100,
          models={"candidate": str(model)}, cohorts={"candidate": [str(root / "candidate-validation")]},
          all_validation_pairs=[["scene", "fixed_keep"]]))
    write(model / "report.json", dict(status="RUNNING", updates=110))
    (model / "updates.jsonl").write_text('{"update":111}\n{"update":112')
    d = data.snapshot(root, tmp_path)
    assert d["total_extra"] == 11 and d["total_target"] == 20
    assert d["evaluations"][0]["state"] == "PARTIAL / WAITING"
    assert d["evaluations"][0]["summary"]["success_rate"] is None


def fixture_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    preset = tmp_path / "outputs/preset"
    source = tmp_path / "outputs/source"
    checkpoint = tmp_path / "outputs/checkpoint"
    manifest = tmp_path / "manifest.json"
    write(source / "report.json", dict(status="COMPLETE", updates=1000, inherited_online_updates=0,
          inherited_self_warmup=5000, new_transitions=1000,
          scope="OWN_EARLY_MOVING_REFRESH_SAC_COMPARISON", early_moving_refresh={}))
    for name in ("online-learner.msgpack", "replay.npz", "ticks.jsonl", "updates.jsonl",
                 "initial-learner.msgpack", "initial-state-audit.json"):
        (source / name).touch()
    write(checkpoint / "manifest.json", {})
    write(manifest, {})
    write(preset / "decision.json", dict(scope="MANUAL_REFRESH_CONTINUATION", seed=0,
          models={"candidate1000": str(source)}, cohorts={"candidate1000": []}, all_validation_pairs=[],
          jobs=[dict(name="candidate-train10000", command=["DO_NOT_EXECUTE", "--checkpoint", str(checkpoint),
                                                         "--manifest", str(manifest)])]))
    return preset, tmp_path / "outputs/new_run"


def test_plan_only_builds_allowlisted_commands_without_output(tmp_path, monkeypatch):
    preset, output = fixture_plan(tmp_path, monkeypatch)
    p = runner.build_plan(preset, output, "candidate", 1200, sys.executable)
    assert not output.exists()
    assert p["inherited_updates_per_arm"] == 1000
    assert len(p["jobs"]) == 3
    assert all(j["command"][0] == sys.executable for j in p["jobs"])
    assert "DO_NOT_EXECUTE" not in str(p)


@pytest.mark.parametrize("updates", [1000, 10001, -1])
def test_budget_restrictions_preserved(tmp_path, monkeypatch, updates):
    preset, output = fixture_plan(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        runner.build_plan(preset, output, "candidate", updates, sys.executable)


def test_refuses_output_overwrite(tmp_path, monkeypatch):
    preset, output = fixture_plan(tmp_path, monkeypatch)
    output.mkdir()
    with pytest.raises(ValueError):
        runner.build_plan(preset, output, "candidate", 1200, sys.executable)


def test_refuses_live_job_before_creating_run(tmp_path, monkeypatch):
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "busy_runs", lambda _: ["existing_live_run"])
    output = tmp_path / "outputs/new"
    with pytest.raises(ValueError, match="existing_live_run"):
        runner.execute(dict(output=str(output)))
    assert not output.exists()


def test_busy_detection_includes_live_child_with_dead_parent(tmp_path, monkeypatch):
    write(tmp_path / "run/status.json", dict(pid=1, jobs=[dict(pid=2, name="train")]))
    monkeypatch.setattr(runner, "live", lambda row: row.get("pid") == 2)
    assert runner.busy_runs(tmp_path) == [str(tmp_path / "run") + " / train"]


def test_failure_stops_pipeline_and_records_exit(tmp_path, monkeypatch):
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "busy_runs", lambda _: [])
    output = tmp_path / "outputs/new"
    marker = tmp_path / "must_not_exist"
    plan = dict(output=str(output), jobs=[dict(name="first", command=[sys.executable, "-c", "raise SystemExit(7)"]),
        dict(name="second", command=[sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"] )])
    with pytest.raises(RuntimeError, match="exit 7"):
        runner.execute(plan)
    status = data.read_json(output / "status.json")
    assert status["status"] == "FAILED" and status["jobs"][0]["exit_code"] == 7
    assert not marker.exists()


def test_completed_process_is_pending_review_not_policy_success(tmp_path, monkeypatch):
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "busy_runs", lambda _: [])
    output = tmp_path / "outputs/new"
    runner.execute(dict(output=str(output), jobs=[dict(name="noop", command=[sys.executable, "-c", "pass"])]))
    assert data.read_json(output / "status.json")["status"] == "COMPLETE_PENDING_REVIEW"


def test_terminal_stop_terminates_child_and_records_stopped(tmp_path):
    (tmp_path / "outputs").mkdir()
    output = tmp_path / "outputs/new"
    code = "\n".join([
        "from pathlib import Path", "import sys", "from scripts import sac_experiment as r",
        f"r.REPO=Path({str(tmp_path)!r})", "r.busy_runs=lambda _: []",
        f"plan=dict(output={str(output)!r}, jobs=[dict(name='wait',command=[sys.executable,'-c','import time; time.sleep(60)'])])",
        "try: r.execute(plan)", "except KeyboardInterrupt: sys.exit(130)",
    ])
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])})
    try:
        deadline = time.monotonic() + 8
        job = None
        while time.monotonic() < deadline:
            jobs = data.read_json(output / "status.json").get("jobs", [])
            if jobs and jobs[0].get("pid"):
                job = jobs[0]
                break
            time.sleep(.03)
        assert job is not None
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=12) == 130
        assert data.read_json(output / "status.json")["status"] == "STOPPED"
        assert not runner.live(job)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_http_is_read_only_and_restricts_files(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (tmp_path / "secret.json").write_text('{"private":true}')
    (outputs / "model.msgpack").write_bytes(b"weights")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, outputs=outputs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert json.load(urlopen(base + "/api/runs")) == {"runs": []}
        for suffix in ["/media?path=../secret.json", "/media?path=model.msgpack", "/api/run?id=../../x"]:
            with pytest.raises(HTTPError) as error:
                urlopen(base + suffix)
            assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/run", data=b"{}", method="POST"))
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
