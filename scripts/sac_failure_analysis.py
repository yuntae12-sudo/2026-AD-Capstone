"""Frozen online traces, explicit denominators, and non-causal failure evidence."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def freeze_trace(run, output):
    """Copy only complete lines from one byte prefix; never touch the live writer."""
    run, output = Path(run), Path(output)
    path = run / "ticks.jsonl"
    limit = path.stat().st_size
    digest = hashlib.sha256()
    consumed = committed = count = 0
    with path.open("rb") as src, (output / "trace.snapshot.jsonl").open("xb") as dst:
        while consumed < limit:
            line = src.readline(limit - consumed)
            if not line:
                raise ValueError(
                    "live trace shrank during snapshot; retry in a new output"
                )
            consumed += len(line)
            if not line.endswith(b"\n"):
                break
            json.loads(line)
            dst.write(line)
            digest.update(line)
            committed += len(line)
            count += 1
    raw = (run / "report.json").read_bytes()
    (output / "run.snapshot.json").write_bytes(raw)
    return json.loads(raw), dict(
        source=str(path.resolve()),
        observed_bytes=limit,
        snapshot_bytes=committed,
        incomplete_tail_bytes=limit - committed,
        lines=count,
        sha256=digest.hexdigest(),
        metadata_sha256=hashlib.sha256(raw).hexdigest(),
    )


def speed_bucket(speed):
    if not math.isfinite(speed) or speed < 0:
        raise ValueError("invalid initial speed")
    if speed < 0.1:
        return "stationary [0,0.1)"
    if speed < 2:
        return "low [0.1,2)"
    if speed < 10:
        return "medium [2,10)"
    return "high [10,+inf)"


def longest_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def terminal_run(values):
    count = 0
    for value in reversed(values):
        if not value:
            break
        count += 1
    return count


def episode_groups(path):
    pending, seen = [], set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if pending and row["episode_uid"] != pending[0]["episode_uid"]:
                if not (pending[-1]["terminated"] or pending[-1]["truncated"]):
                    raise ValueError("episode changed without termination")
                yield pending
                pending = []
            if not pending:
                if row["episode_uid"] in seen:
                    raise ValueError("repeated episode UID")
                seen.add(row["episode_uid"])
            if row["tick"] != len(pending) + 1:
                raise ValueError("noncontiguous episode ticks")
            if pending and (pending[-1]["terminated"] or pending[-1]["truncated"]):
                raise ValueError("trace continues after terminal")
            if pending and (row["scene_id"], row["provider"]) != (
                pending[0]["scene_id"],
                pending[0]["provider"],
            ):
                raise ValueError("episode identity changed")
            if row["provider"] not in ("fixed_keep", "fixed_follow"):
                raise ValueError("unsupported behavior")
            if row["raw_proposal"]["generated_tick"] != row["tick"] - 1:
                raise ValueError("proposal tick mismatch")
            if not math.isfinite(row["reward"]):
                raise ValueError("nonfinite reward")
            pending.append(row)
    if pending:
        yield pending


def summarize_episode(rows, ordinal, metadata, valid_before):
    first, last = rows[0], rows[-1]
    initial = math.hypot(*first["raw_proposal"]["anchor_velocity_world"])
    ended = last["terminated"] or last["truncated"]
    outcome = last["terminal_reason"] if ended else "IN_PROGRESS"
    valid = [r for r in rows if r["sample_valid"]]
    reject = [not r["check_valid"] for r in rows]
    brake = [r["source"] in ("BRAKE", "EMERGENCY") for r in rows]
    # Next tick's measured planning anchor IS this tick's post-step ego state.
    # The final tick has no such anchor; never replace it with its pre-step speed.
    post_speeds = {
        r["tick"]: math.hypot(*n["raw_proposal"]["anchor_velocity_world"])
        for r, n in zip(rows, rows[1:])
        if r["sample_valid"]
    }
    stationary = sum(v < 0.1 for v in post_speeds.values())
    observed = len(post_speeds)
    stationary_source = "next_tick_measured_anchor_partial"
    recorded = metadata.get("episodes", [])
    if ended and ordinal < len(recorded):
        candidate = recorded[ordinal]
        if (
            (
                candidate["scene_id"],
                candidate["provider"],
                candidate["ticks"],
                candidate["outcome"],
            )
            != (first["scene_id"], first["provider"], len(rows), outcome)
            or candidate["accepted"] != sum(r["check_valid"] for r in rows)
            or abs(candidate["return_training"] - sum(r["reward"] for r in rows)) > 1e-8
        ):
            raise ValueError("trace/report episode mismatch")
        reported = candidate.get("stationary_ticks")
        if reported is not None:
            if not stationary <= reported <= stationary + len(valid) - observed:
                raise ValueError("reported post-step stationarity contradicts trace")
            stationary, observed = reported, len(valid)
            stationary_source = "matched_episode_report_actual_post_state"
    warmup = metadata["warmup_transitions"]
    end_valid = valid_before + len(valid)
    phase = (
        "WARMUP"
        if end_valid <= warmup
        else "SAC" if valid_before >= warmup else "MIXED"
    )
    tail = []
    for r in rows[-30:]:
        tail.append(
            dict(
                tick=r["tick"],
                time_s=r["tick"] * 0.1,
                source=r["source"],
                reason=r["check_reason"],
                accepted=r["check_valid"],
                pre_speed=math.hypot(*r["raw_proposal"]["anchor_velocity_world"]),
                post_speed=post_speeds.get(r["tick"]),
                acceleration=r["actual_control"]["acceleration_mps2"],
                reward=r["reward"],
                events=r["event_flags"],
            )
        )
    return dict(
        episode_uid=first["episode_uid"],
        ordinal=ordinal,
        scene_id=first["scene_id"],
        behavior=first["provider"].removeprefix("fixed_").upper(),
        provider=first["provider"],
        phase=phase,
        initial_speed=initial,
        speed_bucket=speed_bucket(initial),
        ticks=len(rows),
        valid_ticks=len(valid),
        completed=bool(ended),
        outcome=outcome,
        collision=any("COLLISION" in r["event_flags"] for r in rows)
        or outcome == "COLLISION",
        offroad=any("OFFROAD" in r["event_flags"] for r in rows),
        integrity_valid=all(r["sample_valid"] for r in rows)
        and outcome != "DATA_CORRUPTION",
        rejected=sum(reject),
        braking=sum(brake),
        stationary_ticks=stationary,
        post_speed_observed_ticks=observed,
        stationary_source=stationary_source,
        return_training=sum(r["reward"] for r in rows),
        first_rejection_tick=next(
            (r["tick"] for r in rows if not r["check_valid"]), None
        ),
        first_braking_tick=next(
            (r["tick"] for r in rows if r["source"] in ("BRAKE", "EMERGENCY")), None
        ),
        longest_rejection_streak=longest_run(reject),
        terminal_rejection_streak=terminal_run(reject),
        terminal_braking_streak=terminal_run(brake),
        last_3s_rejected=sum(reject[-30:]),
        last_3s_braking=sum(brake[-30:]),
        reasons=dict(Counter(r["check_reason"] for r in rows)),
        tail=tail,
    )


def aggregate(episodes):
    eligible = [e for e in episodes if e["completed"] and e["integrity_valid"]]
    ticks = sum(e["ticks"] for e in eligible)
    observed = sum(e["post_speed_observed_ticks"] for e in eligible)

    def fraction(a, b):
        return a / b if b else None

    reasons = sum((Counter(e["reasons"]) for e in eligible), Counter())
    return dict(
        episodes_seen=len(episodes),
        episodes=len(eligible),
        in_progress=sum(not e["completed"] for e in episodes),
        corrupt_episodes=sum(not e["integrity_valid"] for e in episodes),
        unique_scenes=len({e["scene_id"] for e in eligible}),
        ticks=ticks,
        success_rate=fraction(
            sum(e["outcome"] == "ROUTE_GOAL" for e in eligible), len(eligible)
        ),
        collision_rate=fraction(sum(e["collision"] for e in eligible), len(eligible)),
        offroad_rate=fraction(sum(e["offroad"] for e in eligible), len(eligible)),
        rejection_rate=fraction(sum(e["rejected"] for e in eligible), ticks),
        braking_rate=fraction(sum(e["braking"] for e in eligible), ticks),
        stationary_fraction=fraction(
            sum(e["stationary_ticks"] for e in eligible), observed
        ),
        post_speed_coverage=fraction(observed, ticks),
        mean_return=fraction(
            sum(e["return_training"] for e in eligible), len(eligible)
        ),
        outcomes=dict(Counter(e["outcome"] for e in eligible)),
        reasons=dict(reasons),
    )


def analyze(path, metadata):
    episodes, valid_count = [], 0
    for i, rows in enumerate(episode_groups(path)):
        ep = summarize_episode(rows, i, metadata, valid_count)
        episodes.append(ep)
        valid_count += ep["valid_ticks"]
    groups = defaultdict(list)
    for ep in episodes:
        groups[(ep["behavior"], ep["speed_bucket"], ep["phase"])].append(ep)
    return dict(
        scope="ONLINE_FAILURE_ANALYSIS_SNAPSHOT",
        qualified_performance=False,
        versions=metadata["versions"],
        seed=metadata["seed"],
        training_mode=metadata["training_mode"],
        warmup_transitions=metadata["warmup_transitions"],
        valid_transitions_in_snapshot=valid_count,
        # The trace may include a transition whose optimizer update has not committed yet.
        updates_at_last_episode_report=metadata["updates"],
        snapshot_phase=(
            "WARMUP" if valid_count <= metadata["warmup_transitions"] else "ONLINE_SAC"
        ),
        summary=aggregate(episodes),
        by_behavior={
            b: aggregate([e for e in episodes if e["behavior"] == b])
            for b in ("KEEP", "FOLLOW")
        },
        by_initial_speed=[
            dict(behavior=b, speed_bucket=s, phase=p, **aggregate(rows))
            for (b, s, p), rows in sorted(groups.items())
        ],
        episodes=episodes,
        limitations=[
            "Training exploration outcomes, not held-out policy performance.",
            "Incomplete and data-corrupt episodes are excluded from outcome/performance denominators and counted separately.",
            "Stationarity uses actual post-step state; missing last states remain missing unless a matching episode summary supplies the count.",
            "Repeated scenes are not independent samples; no causal or unavoidable-collision claim.",
        ],
    )


def select_collisions(episodes, count):
    candidates = [
        e
        for e in episodes
        if e["completed"] and e["integrity_valid"] and e["collision"]
    ]
    chosen = []
    # Predeclared selection: earliest collision per behavior, then new speed strata,
    # then chronological order. This diagnostic subset is not a rate estimator.
    for behavior in ("KEEP", "FOLLOW"):
        found = next((e for e in candidates if e["behavior"] == behavior), None)
        if found is not None and len(chosen) < count:
            chosen.append(found)
    for ep in candidates:
        strata = {(e["behavior"], e["speed_bucket"]) for e in chosen}
        if (
            len(chosen) < count
            and ep not in chosen
            and (ep["behavior"], ep["speed_bucket"]) not in strata
        ):
            chosen.append(ep)
    for ep in candidates:
        if len(chosen) < count and ep not in chosen:
            chosen.append(ep)
    return chosen


def evaluation_trend(paths, versions, seed):
    """Only paired, completed deterministic validation can flag policy regression."""
    import re

    rows, seen, expected = [], set(), None
    for path in sorted(map(Path, paths)):
        report = json.loads(path.read_text())
        match = re.fullmatch(r"PURE_SAC-(\d+)", report.get("policy", ""))
        if (
            not match
            or report.get("status") != "COMPLETE"
            or report.get("qualified") is not True
        ):
            continue
        if (
            report.get("scope") != "VALIDATION_PERFORMANCE"
            or report["versions"] != versions
            or report["seed"] != seed
        ):
            raise ValueError("validation trend provenance mismatch")
        step = int(match[1])
        pairs = [(e["scene_id"], e["provider"]) for e in report["episodes"]]
        if (
            step in seen
            or len(pairs) != len(set(pairs))
            or (expected is not None and set(pairs) != expected)
        ):
            raise ValueError("duplicate step or unpaired validation scenes")
        seen.add(step)
        expected = set(pairs)
        from scripts.sac_experiment_runtime import summarize_episodes

        if summarize_episodes(report["episodes"]) != report["by_behavior"]:
            raise ValueError("validation metrics mismatch")
        rows.append(
            dict(
                step=step,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                by_behavior=report["by_behavior"],
            )
        )
    rows.sort(key=lambda r: r["step"])
    comparisons = []
    for before, after in zip(rows, rows[1:]):
        for behavior in ("KEEP", "FOLLOW"):
            b, a = before["by_behavior"][behavior], after["by_behavior"][behavior]
            reward, stationary = (
                a["mean_return"] - b["mean_return"],
                a["stationary_fraction"] - b["stationary_fraction"],
            )
            comparisons.append(
                dict(
                    before_step=before["step"],
                    after_step=after["step"],
                    behavior=behavior,
                    reward_delta=reward,
                    stationary_delta=stationary,
                    reward_up_and_stationary_up=reward > 1e-9 and stationary > 1e-9,
                )
            )
    return dict(
        status="AVAILABLE" if comparisons else "INSUFFICIENT_PAIRED_EVALUATIONS",
        evaluations=rows,
        comparisons=comparisons,
        interpretation="A reward/stationarity flag calls for inspection; it does not prove reward hacking or improvement.",
    )
