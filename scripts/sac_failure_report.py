"""Human-readable report and plots from frozen failure-analysis evidence."""

from pathlib import Path


def number(value, percent=False):
    if value is None:
        return "N/A"
    return f"{value*100:.1f}%" if percent else f"{value:.3f}"


def write_markdown(report, path):
    s = report["summary"]
    lines = [
        "# 순수 SAC 실패 분석 보고서",
        "",
        f"- 단계: **{report['snapshot_phase']}** / seed {report['seed']}",
        f"- 스냅샷 유효 전이: {report['valid_transitions_in_snapshot']:,}개; 마지막 episode 보고서의 SAC 업데이트: {report['updates_at_last_episode_report']:,}회",
        f"- 완료·정상 데이터 episode: {s['episodes']}개 / 진행 중: {s['in_progress']}개 / 손상 데이터 episode: {s['corrupt_episodes']}개",
        f"- 고유 장면: {s['unique_scenes']}개. 반복 방문은 독립 장면으로 세지 않는다.",
        "",
        "이 보고서는 학습 중 탐색 경험의 실패 진단이다. validation 정책 성능이나 최종 비교 결과가 아니다.",
        "WARMUP은 업데이트 전 초기 정책이다. 충돌 전 제동이 관찰돼도 제동이 충돌의 원인이라거나 회피 불가능했다고 단정하지 않는다.",
        "",
        "## KEEP / FOLLOW 집계",
        "",
        "성공·충돌률과 평균 보상은 완료·정상 episode 기준, 거절·제동률은 그 episode의 tick 기준이다.",
        "정지 비중은 실제 적용 후 속도 <0.1m/s인 상태 기준이며, 마지막 상태가 누락된 경우 관측 커버리지를 함께 표시한다.",
        "",
        "| 행동 | Episode | 성공 | 충돌 | 거절 | 제동 | 정지 | 상태 커버리지 | 평균 보상 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b, r in report["by_behavior"].items():
        vals = [
            number(r[k], True)
            for k in (
                "success_rate",
                "collision_rate",
                "rejection_rate",
                "braking_rate",
                "stationary_fraction",
                "post_speed_coverage",
            )
        ]
        lines.append(
            f"| {b} | {r['episodes']} | "
            + " | ".join(vals)
            + f" | {number(r['mean_return'])} |"
        )
    lines += [
        "",
        "![행동별 실패 지표](metrics.png)",
        "",
        "## 초기 속도별 결과",
        "",
        "| 행동 | 초기 속도 m/s | 단계 | Episode | 충돌률 | 거절률 | 제동률 | 정지 비중 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["by_initial_speed"]:
        vals = [
            number(r[k], True)
            for k in (
                "collision_rate",
                "rejection_rate",
                "braking_rate",
                "stationary_fraction",
            )
        ]
        lines.append(
            f"| {r['behavior']} | {r['speed_bucket']} | {r['phase']} | {r['episodes']} | "
            + " | ".join(vals)
            + " |"
        )
    lines += [
        "",
        "## 경로 거절 이유",
        "",
        "| 행동 | 검사 결과 | Tick 수 |",
        "|---|---|---:|",
    ]
    for b, r in report["by_behavior"].items():
        for reason, count in sorted(r["reasons"].items(), key=lambda x: -x[1]):
            if reason != "OK":
                lines.append(f"| {b} | {reason} | {count} |")
    collisions = [
        e
        for e in report["episodes"]
        if e["collision"] and e["completed"] and e["integrity_valid"]
    ]
    lines += [
        "",
        "## 충돌 직전 경로 거절·제동",
        "",
        "아래 연속 길이는 충돌 tick까지 이어진 구간이다. 제동은 BRAKE/EMERGENCY 실행 source를 뜻한다.",
        "",
        "| Episode | 장면 | 행동 | 초기 속도 | 최초 거절 tick | 최초 제동 tick | 직전 연속 거절 | 직전 연속 제동 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for e in collisions:
        lines.append(
            f"| {e['ordinal']} | {e['scene_id']} | {e['behavior']} | {e['initial_speed']:.2f} | {e['first_rejection_tick']} | {e['first_braking_tick']} | {e['terminal_rejection_streak']} | {e['terminal_braking_streak']} |"
        )
    lines += [
        "",
        "## 실제 Waymax 재현으로 확인한 충돌 사례",
        "",
        "선정 규칙: 행동별 최초 충돌 → 새로운 행동·속도 구간 → 시간순. 아래 소수 사례로 전체 충돌 원인의 비율을 추정하지 않는다.",
        "저장된 raw 경로를 다시 입력하고 제어·검사·사건·보상 및 다음 tick의 측정 앵커를 대조했다.",
        "원본에는 종료 직후 상태가 없으므로 최종 상대 위치는 대조를 통과한 제어로 재현한 값이다.",
    ]
    eps = {e["episode_uid"]: e for e in report["episodes"]}
    for case in report["replays"]:
        e = eps[case["episode_uid"]]
        lines += [
            "",
            f"### Episode {e['ordinal']} / {e['behavior']} / {e['scene_id']}",
            "",
        ]
        if case["status"] != "VERIFIED":
            lines.append(
                f"재현 대조 실패: `{case['error']}`. 충돌 상대를 판정하지 않는다."
            )
            continue
        lines += [
            f"- {case['verified_ticks']}tick 제어·검사·종료 대조 통과, {case['verified_post_anchors']}개 post-state 앵커 대조 통과.",
            f"- 종료 속도: {case['final_ego_speed']:.3f}m/s. 관찰된 순서: `{case['chronological_pattern']}`.",
        ]
        for c in case["contacts"]:
            f = c["final"]
            lines.append(
                f"- 충돌 객체 {c['object_id']}: 접근 방향 **{c['approach_side']}**, 종료 시 상대 종방향 {f['relative_longitudinal_m']:.2f}m / 횡방향 {f['relative_lateral_m']:.2f}m, 상대 차량 속도 {f['other_speed']:.2f}m/s."
            )
        lines += [
            "",
            "| Tick | 적용 전 속도 | 실행 source | 경로 검사 | 적용 가속도 |",
            "|---:|---:|---|---|---:|",
        ]
        for t in e["tail"][-10:]:
            lines.append(
                f"| {t['tick']} | {t['pre_speed']:.3f} | {t['source']} | {t['reason']} | {t['acceleration']:.3f} |"
            )
    if any(c["status"] == "VERIFIED" and c["contacts"] for c in report["replays"]):
        lines += ["", "![충돌 전 속도·상대 위치](collision-timelines.png)"]
    trend = report["evaluation_trend"]
    lines += ["", "## 보상 상승과 정지 비중 증가 점검", ""]
    if not trend["comparisons"]:
        lines.append(
            "**아직 판단 불가.** 같은 validation 장면에서 완료된 서로 다른 학습 시점의 평가가 2개 이상 필요하다. 초기 수집의 장면 구성 차이를 학습 향상으로 해석하지 않는다."
        )
    else:
        lines += [
            "| 이전→이후 전이 수 | 행동 | 보상 변화 | 정지 비중 변화 | 동시 증가 점검 |",
            "|---|---|---:|---:|---|",
        ]
        for r in trend["comparisons"]:
            lines.append(
                f"| {r['before_step']}→{r['after_step']} | {r['behavior']} | {r['reward_delta']:.4f} | {r['stationary_delta']*100:.2f}%p | {'검토 필요' if r['reward_up_and_stationary_up'] else '해당 없음'} |"
            )
    lines += [
        "",
        "## 재현 자료와 한계",
        "",
        "- `analysis.json`: 전체 episode·집계·선정 기준·재현 결과·평가 추이.",
        "- `trace.snapshot.jsonl`, `run.snapshot.json`: 분석 시점에 동결한 원본. 진행 중 로그의 마지막 미완성 줄은 제외한다.",
        f"- trace SHA256: `{report['snapshot']['sha256']}` / 완전한 줄 {report['snapshot']['lines']:,}개.",
        "- 주변 차량은 WOMD 로그를 따라가며 자차 제동에 반응하지 않는다. 후방 접촉이 관찰돼도 다른 정책으로 회피할 수 없었다는 뜻은 아니다.",
        "- 학습 데이터·가중치·검사 기준·진행 중인 평가 조건은 변경하지 않았다.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    keys = ("collision_rate", "rejection_rate", "braking_rate", "stationary_fraction")
    for ax, (behavior, row) in zip(axes, report["by_behavior"].items()):
        values = [row[k] for k in keys]
        bars = ax.bar(
            ["Collision", "Rejected", "Braking", "Stationary"],
            [v or 0 for v in values],
            color=["#d85b5b", "#d69b3b", "#8f74b6", "#348e86"],
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                "N/A" if v is None else f"{v:.1%}",
                ha="center",
                fontsize=9,
            )
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{behavior} | {row['episodes']} complete episodes")
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Fraction (collision: episodes; other metrics: ticks)")
    fig.suptitle(
        f"{report['snapshot_phase']} / training diagnostics, not validation performance"
    )
    fig.savefig(output / "metrics.png", dpi=150)
    plt.close(fig)
    cases = [
        c for c in report["replays"] if c["status"] == "VERIFIED" and c["contacts"]
    ]
    if not cases:
        return
    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(12, 3.4 * len(cases)),
        squeeze=False,
        layout="constrained",
    )
    episodes = {e["episode_uid"]: e for e in report["episodes"]}
    for (left, right), case in zip(axes, cases):
        ep = episodes[case["episode_uid"]]
        track = case["contacts"][0]["last_3s"]
        left.plot(
            [r["time_s"] for r in track],
            [r["ego_speed"] for r in track],
            label="Ego",
            color="#2465ba",
        )
        for contact_index, c in enumerate(case["contacts"]):
            t = c["last_3s"]
            color = ("#d67c27", "#8b62a8", "#35846d")[contact_index % 3]
            left.plot(
                [r["time_s"] for r in t],
                [r["other_speed"] for r in t],
                label=f"Object {c['object_id']}",
                color=color,
            )
            right.plot(
                [r["time_s"] for r in t],
                [r["relative_longitudinal_m"] for r in t],
                label=f"Object {c['object_id']}",
                color=color,
            )
        for t in ep["tail"]:
            if t["source"] in ("BRAKE", "EMERGENCY"):
                left.axvspan(
                    t["time_s"] - 0.1, t["time_s"], color="#d85b5b", alpha=0.12
                )
        left.set_title(f"Episode {ep['ordinal']} / {ep['behavior']} (red: braking)")
        left.set_ylabel("Speed [m/s]")
        right.set_title("Relative center: +front / -rear")
        right.set_ylabel("Longitudinal distance [m]")
        right.axhline(0, color="gray", lw=0.7)
        for ax in (left, right):
            ax.axvline(
                case["verified_ticks"] * 0.1,
                color="#ac3b48",
                ls="--",
                lw=1,
                label="Collision tick",
            )
            ax.set_xlabel("Simulation time [s]")
            ax.grid(alpha=0.15)
            ax.legend(fontsize=8)
    fig.savefig(output / "collision-timelines.png", dpi=150)
    plt.close(fig)
