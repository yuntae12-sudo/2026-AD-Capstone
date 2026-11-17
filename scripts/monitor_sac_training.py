#!/usr/bin/env python3
"""Read-only terminal dashboard for the complete multi-seed SAC experiment.

Uses only the Python standard library. Ctrl+C stops this viewer, not training.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/pure_sac_stable_20261117"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def progress(count, budget, width):
    ratio = min(1.0, max(0.0, count / budget)) if budget else 0.0
    filled = int(ratio * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {ratio:6.1%}"


def latest_update(path):
    """Read a bounded tail, excluding an unfinished JSONL write."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            lines = stream.read().split(b"\n")[:-1]
    except FileNotFoundError:
        return {}
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeError):
            continue
    return {}


def render(output, job, width=26):
    phases = job.get("phases", {})
    protocol = job.get("protocol", {})
    seeds = protocol.get("seeds", [0, 1, 2])
    budget = job["budget_per_seed"]
    warmup = protocol.get("warmup_transitions", 5000)
    rows, total = [], 0
    for seed in seeds:
        directory = output / f"online-seed{seed}"
        report = read_json(directory / "report.json")
        online = phases.get(f"online-seed{seed}", {})
        final = phases.get(f"pure-sac-seed{seed}", {})
        count = report.get("valid_online_transitions", 0)
        updates = report.get("updates", 0)
        status = report.get("status", "WAITING")
        if status == "RUNNING" and online.get("status") == "RUNNING":
            latest = latest_update(directory / "updates.jsonl")
            if latest.get("step", 0) > count:
                count = latest["step"]
                updates = max(updates, count - warmup)
        total += min(count, budget)
        label = "대기"
        if report:
            label = "초기 경험 수집" if count < warmup else "SAC 학습"
        if status == "COMPLETE":
            label = "학습 완료 / 최종 평가 대기"
        if final.get("status") == "RUNNING":
            label = "최종 평가 중"
        elif final.get("status") == "COMPLETE":
            label = "학습·평가 완료"
        elif online.get("status") == "RUNNING" and status == "RUNNING":
            evaluations = sorted((directory / "evaluations").glob("step-*.json"))
            if evaluations:
                evaluation = read_json(evaluations[-1])
                if evaluation.get("status") == "RUNNING":
                    label = f"중간 평가 중 ({len(evaluation.get('episodes', []))}개 장면 완료)"
        if status == "ERROR" or "FAILED" in (online.get("status"), final.get("status")):
            label = "오류로 중단"
        rows.extend(
            [
                f"seed {seed}  {progress(count, budget, width)}  {count:,} / {budget:,}",
                f"        {label} | SAC 업데이트 {updates:,}회",
            ]
        )
        if report.get("error"):
            rows.append(f"        오류: {report['error']}")
    status = job["status"]
    labels = {
        "RUNNING": "진행 중",
        "COMPLETE": "전체 학습·평가·비교 완료",
        "STOPPED_FOR_REVIEW": "오류 또는 중단 — 로그 확인 필요",
    }
    active = [
        name for name, phase in phases.items() if phase.get("status") == "RUNNING"
    ]
    lines = [
        f"SAC 실험 진행 현황 | {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"실험: {output.name}",
        f"상태: {labels.get(status, status)}",
        f"현재 단계: {', '.join(active) if status == 'RUNNING' and active else '-'}",
        "",
        f"전체 수집 목표 {progress(total, budget * len(seeds), width)}  {total:,} / {budget * len(seeds):,}",
        "",
        *rows,
        "",
        "진행률 = 수집량 기준. 평가 시간은 포함하지 않음.",
        "seed별 초과 수집은 전체 진행률에 100%까지만 반영.",
        "수집 수치는 에피소드 종료 시, 학습 수치는 업데이트 로그로 갱신.",
        "수집 100% 이후에도 평가·비교가 남을 수 있음. 완료는 성능 합격을 뜻하지 않음.",
        "Ctrl+C: 화면만 종료 (학습은 계속 실행)",
    ]
    if job.get("error"):
        lines.extend(["", f"오류: {job['error']}", f"로그: {output / 'logs'}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument(
        "--once", action="store_true", help="Print one snapshot and exit"
    )
    args = parser.parse_args()
    if not 0 < args.interval <= 60:
        parser.error("--interval must be greater than 0 and at most 60 seconds")
    output = args.output.resolve()
    if not (output / "job.json").exists():
        parser.error(f"experiment job.json not found: {output}")
    try:
        while True:
            try:
                job = read_json(output / "job.json")
                if not job:
                    raise ValueError("job.json을 읽을 수 없음")
                width = max(10, min(30, shutil.get_terminal_size().columns - 52))
                screen = render(output, job, width)
                stopped = False
                if job["status"] == "RUNNING" and job.get("pid"):
                    try:
                        os.kill(int(job["pid"]), 0)
                    except ProcessLookupError:
                        stopped = True
                        screen += "\n\n실행 프로세스가 없음. 마지막 보고서만 표시 중 — 로그 확인 필요."
                    except PermissionError:
                        pass
                if sys.stdout.isatty() and not args.once:
                    print("\033[2J\033[H", end="")
                print(screen, flush=True)
                if job["status"] == "COMPLETE":
                    return 0
                if stopped or job["status"] in ("STOPPED_FOR_REVIEW", "ERROR"):
                    return 1
                if args.once:
                    return 0
            except (OSError, ValueError, KeyError) as exc:
                print(f"보고서 읽기 실패: {exc} (다음 주기에 재확인)", flush=True)
                if args.once:
                    return 2
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n모니터를 종료했습니다. 학습은 계속 실행됩니다.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
