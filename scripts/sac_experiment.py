#!/usr/bin/env python3
"""Plan/run an existing pure SAC continuation recipe from the terminal."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.sac_dashboard_data import SCOPES, read_json
from scripts.monitor_sac_continuation import live


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19])


def build_plan(preset, output, arm, updates, python):
    """Read paths from the preset; never execute commands provided by that file."""
    preset, output = Path(preset).resolve(), Path(output).resolve()
    d = read_json(preset / "decision.json")
    if d.get("scope") not in SCOPES:
        raise ValueError("지원하는 continuation 실험 폴더를 --preset으로 지정하세요.")
    if output.exists() or not output.is_relative_to(REPO / "outputs"):
        raise ValueError("--output은 outputs 아래 아직 존재하지 않는 새 폴더여야 합니다.")
    arms = ["candidate", "control"] if arm == "both" else [arm]
    models, cohorts, sources, jobs, inherited_values = {}, {}, {}, [], set()
    for name in arms:
        source = Path(d["models"][name + "1000"]).resolve()
        report = read_json(source / "report.json")
        # Use the existing budget validation without importing training libraries.
        from scripts.continue_sac_refresh_context import validate_budget
        validate_budget(report, updates)
        inherited_values.add(report["updates"])
        old = next(j["command"] for j in d["jobs"] if j["name"].startswith(name + "-train"))
        def argument(flag):
            return str(Path(old[old.index(flag) + 1]).resolve())
        checkpoint, manifest = argument("--checkpoint"), argument("--manifest")
        if not Path(checkpoint, "manifest.json").is_file() or not Path(manifest).is_file():
            raise ValueError("원본 checkpoint/manifest가 없습니다.")
        for file in ("online-learner.msgpack", "replay.npz", "ticks.jsonl", "updates.jsonl",
                     "initial-learner.msgpack", "initial-state-audit.json"):
            if not (source / file).is_file():
                raise ValueError(f"원본 파일 누락: {source / file}")
        model = output / f"{name}{updates}"
        evaluation = output / f"{name}-validation"
        models.update({name: str(model), name + "1000": str(source)})
        cohorts[name] = [str(evaluation)]
        cohorts[name + "1000"] = d["cohorts"][name + "1000"]
        sources[name] = str(source)
        def job(label, script, args):
            return dict(name=label, command=[python, str(REPO / "scripts" / script), *args])
        jobs.extend([
            job(f"{name}-train{updates}", "continue_sac_refresh_context.py",
                ["--source", str(source), "--checkpoint", checkpoint, "--manifest", manifest,
                 "--output", str(model), "--updates", str(updates)]),
            job(f"{name}-audit{updates}", "audit_sac_refresh_context_continuation.py",
                ["--directory", str(model), "--checkpoint", checkpoint, "--manifest", manifest,
                 "--output", str(output / f"{name}-audit{updates}.json")]),
            job(f"{name}-validation", "evaluate_sac_short_updates.py",
                ["--diagnostic", str(model), "--manifest", manifest, "--output", str(evaluation),
                 "--variant", "online", "--labels", "updated"]),
        ])
    if len(inherited_values) != 1:
        raise ValueError("두 arm의 시작 업데이트 수가 다릅니다.")
    return dict(scope="MANUAL_REFRESH_CONTINUATION", created_kst=stamp(), seed=d["seed"],
                active_arms=arms, preset=str(preset), output=str(output), sources=sources,
                inherited_updates_per_arm=inherited_values.pop(), total_updates_per_arm=updates,
                models=models, cohorts=cohorts, jobs=jobs, all_validation_pairs=d["all_validation_pairs"],
                rule_reference=d.get("rule_reference"),
                note="Original completed fresh source, not the latest 10000 checkpoint. BC0/teacher0; no automatic budget extension or adoption.")


def busy_runs(outputs):
    busy = []
    for path in Path(outputs).glob("*/status.json"):
        data = read_json(path)
        if live(data):
            busy.append(str(path.parent))
        for job in data.get("jobs", []):
            if live(job):
                busy.append(str(path.parent) + " / " + job.get("name", "child"))
        for worker in path.parent.glob("*/status.json"):
            state = read_json(worker)
            if live(state):
                busy.append(str(worker.parent))
    return sorted(set(busy))


def execute(plan):
    output = Path(plan["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Serializes this launcher's own runs. Also checks older experiment runners.
    with (REPO / "outputs/.sac-manual-run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("다른 수동 실행이 진행 중입니다.")
        busy = busy_runs(REPO / "outputs")
        if busy:
            raise ValueError("기존 실험/후속 작업이 실행 중입니다. 완료 후 실행하세요:\n" + "\n".join(busy))
        output.mkdir(exist_ok=False)
        # Record the exact source files for this run; training/audit perform deeper checks.
        plan["launcher_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        atomic_json(output / "decision.json", plan)
        status = dict(status="RUNNING", started_kst=stamp(), jobs=[], **process_identity(os.getpid()))
        path = output / "status.json"
        atomic_json(path, status)
        child = None
        job = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            for spec in plan["jobs"]:
                child = None
                job = dict(spec, status="RUNNING", started_kst=stamp())
                status["jobs"].append(job)
                print("\n▶ " + spec["name"], flush=True)
                print(shlex.join(spec["command"]), flush=True)
                with (output / (spec["name"] + ".log")).open("w") as log:
                    child = subprocess.Popen(spec["command"], cwd=REPO, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        job.update(process_identity(child.pid))
                    except FileNotFoundError:
                        pass  # An immediately failing child still has an exit status.
                    atomic_json(path, status)
                    code = child.wait()
                job.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code, finished_kst=stamp())
                atomic_json(path, status)
                if code:
                    raise RuntimeError(f"{spec['name']} 실패(exit {code}). {output / (spec['name'] + '.log')}")
            status["status"] = "COMPLETE_PENDING_REVIEW"
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            status["status"] = "STOPPED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            status["error"] = str(exc) or "사용자 중단"
            if job and job["status"] == "RUNNING":
                job.update(status=status["status"], exit_code=child.returncode if child else None)
            raise
        finally:
            status["finished_kst"] = stamp()
            atomic_json(path, status)
            signal.signal(signal.SIGTERM, previous)
        print("학습·감사·평가 실행 완료. 결과 검토가 필요합니다: " + str(output), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "run"])
    p.add_argument("--preset", type=Path, required=True, help="기존 matched 실험 폴더")
    p.add_argument("--output", type=Path, required=True, help="새 결과 폴더")
    p.add_argument("--arm", choices=["both", "candidate", "control"], default="both")
    p.add_argument("--updates", type=int, default=10000, help="기존1,000 포함 총 update 수, 최대10,000")
    args = p.parse_args()
    os.chdir(REPO)
    try:
        plan = build_plan(args.preset, args.output, args.arm, args.updates, sys.executable)
        print(f"시작: 기존 {plan['inherited_updates_per_arm']:,} → 총 {args.updates:,}회 / seed {plan['seed']} / {args.arm}")
        print("최근10,000 모델의 추가 연장이 아니라 원본1,000 모델에서 시작합니다.")
        for job in plan["jobs"]:
            print(job["name"] + ": " + shlex.join(job["command"]))
        if args.action == "run":
            execute(plan)
        else:
            print("계획 확인만 완료. 학습·출력 폴더 생성 없음. 실행하려면 plan을 run으로 바꾸세요.")
    except KeyboardInterrupt:
        print("중단됨. 최종 저장 전 중단한 학습은 이 도구로 재개할 수 없습니다.", file=sys.stderr)
        return 130
    except (ValueError, KeyError, StopIteration, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
