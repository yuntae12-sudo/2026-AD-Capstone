#!/usr/bin/env python3
"""Read-only progress viewer for a matched SAC continuation experiment."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # Reports may be observed between writes. Never modify or restart jobs.
        return {}


def latest_update(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            complete = stream.read().split(b"\n")[:-1]
    except FileNotFoundError:
        return 0
    for line in reversed(complete):
        try:
            return int(json.loads(line).get("update", 0))
        except (ValueError, UnicodeError):
            continue
    return 0


def live(job):
    if not job.get("pid") or not job.get("start_ticks"):
        return False
    try:
        fields = Path(f'/proc/{int(job["pid"])}/stat').read_text().split(") ", 1)[1].split()
        return fields[0] != "Z" and fields[19] == str(job["start_ticks"])
    except (FileNotFoundError, ProcessLookupError):
        return False


def bar(value, total, width=26):
    ratio = min(1., max(0., value / total)) if total else 0.
    filled = int(ratio * width)
    return f"[{'#' * filled}{'-' * (width-filled)}] {ratio:6.1%}"


def render(root):
    decision, status = read_json(root / "decision.json"), read_json(root / "status.json")
    if not decision or not status:
        return "실험 설정 또는 상태를 읽는 중 — 다음 갱신에서 다시 확인합니다."
    if decision.get("scope") != "MATCHED_CONTINUATION_1000_TO_10000_OWN_REFRESH":
        raise ValueError("this viewer requires the matched context-continuation decision")
    inherited, total = decision["inherited_updates_per_arm"], decision["total_updates_per_arm"]
    target = total - inherited
    jobs = status.get("jobs", [])
    current = [job for job in jobs if job["status"] == "RUNNING"]
    state = {"RUNNING": "진행 중", "FAILED": "오류 — 로그 확인 필요",
             "COMPLETE_PENDING_REVIEW": "학습·평가 실행 완료 — 결과 검토 필요"}.get(
                 status["status"], status["status"])
    if status["status"] == "RUNNING" and not live(status):
        state = "실행 프로세스 없음 — 상태 기록 확인 필요"
    lines = [f"SAC 이어 학습 | {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"실험: {root.name}", f"전체 상태: {state}", ""]
    completed = 0
    for arm, label in (("candidate", "새 수집 후보"), ("control", "대조군")):
        directory = Path(decision["models"][arm])
        report = read_json(directory / "report.json")
        count = max(inherited, int(report.get("updates", inherited)),
                    latest_update(directory / "updates.jsonl"))
        extra = max(0, count-inherited)
        completed += min(extra, target)
        phase = report.get("status", "WAITING")
        text = {"WAITING": "대기", "RESTORING": "기존 수집 상태 복원 중",
                "RUNNING": "새 자기 경험 수집·SAC 업데이트 중", "COMPLETE": "학습 완료",
                "FAILED": "학습 오류"}.get(phase, phase)
        train = next((j for j in jobs if j["name"] == f"{arm}-train{total}"), {})
        if train.get("status") == "RUNNING" and not live(train):
            text += " · 프로세스 종료 여부 확인 필요"
        lines += [f"{label}: {bar(extra,target)}  추가 {extra:,}/{target:,}",
                  f"  {text} | 누적 업데이트 {count:,}/{total:,} (기존 {inherited:,} 포함)"]
        if phase == "RESTORING":
            restored = read_json(directory / "restoration-progress.json")
            n, budget = restored.get("restored", 0), restored.get("total", inherited)
            lines.append(f"  상태 복원: {bar(n,budget)} {n:,}/{budget:,} — 새 학습으로 세지 않음")
        if report.get("error"):
            lines.append(f"  오류: {report['error']}")
        evaluation = read_json(root / f"{arm}-validation" / "report.json")
        if evaluation:
            lines.append(f"  개발 평가: {len(evaluation.get('episodes',[]))}/{len(decision['all_validation_pairs'])}회 · {evaluation['status']}")
    lines += ["", f"전체 추가 학습: {bar(completed,2*target)} {completed:,}/{2*target:,}",
              "현재 작업: " + (", ".join(j["name"] for j in current) or "-"),
              "진행률은 새 학습량 기준이며 복원·평가 시간은 포함하지 않습니다.",
              "학습 100%는 안정적인 주행 성능을 뜻하지 않습니다.",
              "Ctrl+C는 화면만 종료하며 학습은 계속됩니다."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < .5:
        parser.error("interval must be at least 0.5 seconds")
    try:
        while True:
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(render(args.output), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
