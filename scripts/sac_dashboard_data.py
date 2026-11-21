"""Read-only adapters for local SAC experiment artifacts (standard library only)."""
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path

from scripts.monitor_sac_continuation import latest_update, live

SCOPES = {"MATCHED_CONTINUATION_1000_TO_10000_OWN_REFRESH", "MANUAL_REFRESH_CONTINUATION"}


def read_json(path):
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("path outside outputs directory")
    return path


def experiments(outputs):
    return sorted((p for p in Path(outputs).iterdir() if p.is_dir()
                   and not p.is_symlink() and read_json(p / "decision.json").get("scope") in SCOPES),
                  key=lambda p: p.name, reverse=True)


def summarize(episodes):
    ticks = sum(e["ticks"] for e in episodes)
    outcomes = Counter(e["outcome"] for e in episodes)
    return dict(episodes=len(episodes), scenes=len({e["scene_id"] for e in episodes}),
                ticks=ticks, outcomes=dict(outcomes),
                success_rate=outcomes["ROUTE_GOAL"] / len(episodes) if episodes else None,
                collision_rate=outcomes["COLLISION"] / len(episodes) if episodes else None,
                **{k: sum(e[k] for e in episodes) / ticks if ticks else None
                   for k in ("rejected", "braking", "stationary")},
                mean_return=math.fsum(e["return_training"] for e in episodes) / len(episodes)
                if episodes else None)


def curve(path, limit=250):
    """Bucket means over complete JSONL rows; incomplete trailing writes are ignored."""
    keys = ("actor_loss", "q1_loss", "q2_loss", "alpha", "actor_q1_abs_max", "target_abs_max")
    rows = []
    try:
        with Path(path).open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get("update"), (int, float)):
                        rows.append(row)
                except ValueError:
                    continue
    except OSError:
        pass
    width = max(1, math.ceil(len(rows) / limit))
    points = []
    for start in range(0, len(rows), width):
        batch = rows[start:start + width]
        point = {"update": batch[-1]["update"]}
        for key in keys:
            values = [r[key] for r in batch if isinstance(r.get(key), (int, float))
                      and math.isfinite(r[key])]
            point[key] = math.fsum(values) / len(values) if values else None
        points.append(point)
    return dict(points=points, rows=len(rows), bucket_size=width)


def snapshot(root, outputs):
    root = inside(root, outputs)
    decision = read_json(root / "decision.json")
    if decision.get("scope") not in SCOPES:
        raise ValueError("unsupported experiment")
    status = read_json(root / "status.json")
    total = decision["total_updates_per_arm"]
    inherited = decision["inherited_updates_per_arm"]
    arms = decision.get("active_arms", ["candidate", "control"])
    jobs = status.get("jobs", [])
    cards, evaluations, media = [], [], []
    links = []
    for arm in arms:
        model = inside(decision["models"][arm], outputs)
        report = read_json(model / "report.json")
        updates = max(inherited, report.get("updates", inherited), latest_update(model / "updates.jsonl"))
        job = next((j for j in jobs if j.get("name") == f"{arm}-train{total}"), {})
        evaluation = read_json(root / f"{arm}-validation/report.json")
        cards.append(dict(arm=arm, updates=updates, total=total, inherited=inherited,
                          extra=max(0, updates - inherited), target=total-inherited,
                          state=report.get("status", "WAITING"),
                          process_alive=live(job), job_state=job.get("status", "WAITING"),
                          evaluation_count=len(evaluation.get("episodes", [])),
                          evaluation_state=evaluation.get("status", "WAITING"),
                          curve=curve(model / "updates.jsonl"),
                          restoration=read_json(model / "restoration-progress.json")))
    for label, dirs in decision.get("cohorts", {}).items():
        rows, states = [], []
        for directory in dirs:
            directory = inside(directory, outputs)
            report = read_json(directory / "report.json")
            states.append(report.get("status", "WAITING"))
            rows.extend(report.get("episodes", []))
            if report:
                links.append(dict(label=label + " 평가 JSON", path=str((directory / "report.json").relative_to(outputs))))
        evaluations.append(dict(label=label, state="COMPLETE" if states and all(s == "COMPLETE" for s in states) else "PARTIAL / WAITING",
                                reused=label not in arms, summary=summarize(rows),
                                by_provider={p: summarize([e for e in rows if e["provider"] == p])
                                             for p in ("fixed_keep", "fixed_follow")}, episodes=rows))
    rule = decision.get("rule_reference")
    if rule:
        report = read_json(inside(rule, outputs) / "captures/report.json")
        rows = report.get("episodes", [])
        evaluations.append(dict(label="rule_reference", state=report.get("status", "WAITING"), reused=True,
                                summary=summarize(rows), by_provider={p: summarize([e for e in rows if e["provider"] == p])
                                for p in ("fixed_keep", "fixed_follow")}, episodes=rows))
    # Current GIFs and explicitly identified prior candidates are never mixed silently.
    media_roots = [(root / "visualization", "이번 실험")]
    for arm in arms:
        old = decision.get("models", {}).get(arm + "1000")
        if old:
            media_roots.append((inside(old, outputs).parent / "visualization", arm + " · 이전 1,000회"))
    for folder, label in media_roots:
        if folder.exists():
            for path in sorted(folder.rglob("*.gif"))[:40]:
                try:
                    path = inside(path, outputs)
                except ValueError:
                    continue
                media.append(dict(label=label, name=path.name, path=str(path.relative_to(outputs))))
    workers = []
    for name in ("failure-analysis-job", "visualization", "policy-diagnostics", "capture-metrics-job"):
        worker = read_json(root / name / "status.json")
        if worker:
            workers.append(dict(name=name, state=worker.get("status"), alive=live(worker)))
    for name in ("candidate-preliminary-review.json", "full-validation-review.json", "decision.json"):
        if (root / name).is_file():
            links.append(dict(label=name, path=str((root / name).relative_to(outputs))))
    logs = []
    for path in sorted(root.glob("*.log")):
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 6000))
                tail = stream.read().decode("utf-8", errors="replace")
            logs.append(dict(name=path.name, tail=tail))
        except OSError:
            continue
    return dict(name=root.name, updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                state=status.get("status", "UNKNOWN"), process_alive=live(status),
                seed=decision.get("seed"), arms=cards, evaluations=evaluations,
                evaluation_target=len(decision.get("all_validation_pairs", [])),
                jobs=[dict(name=j.get("name"), state=j.get("status"), alive=live(j),
                           exit_code=j.get("exit_code")) for j in jobs], workers=workers,
                media=media, links=links, logs=logs,
                total_extra=sum(min(a["extra"], a["target"]) for a in cards),
                total_target=sum(a["target"] for a in cards))
