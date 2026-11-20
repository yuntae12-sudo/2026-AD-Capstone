"""Protocol metrics from complete KEEP/FOLLOW CapturingEnvironment traces.

The first executed_xy point is the selected trajectory at the post-step time.
Raw proposals are not tracking references; emergency ticks without a selected
trajectory do not contribute a zero error to the RMS denominator.
"""
from collections import Counter
import math


def _xy(value):
    if len(value) != 2 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError("requires finite XY coordinates")
    return tuple(float(x) for x in value)


def episode_metrics(trace):
    if trace["provider"] not in ("fixed_keep", "fixed_follow"):
        raise ValueError("requires fixed KEEP/FOLLOW")
    frames = trace["frames"]
    if len(frames) < 2 or [f["tick"] for f in frames] != list(range(len(frames))):
        raise ValueError("requires a contiguous episode starting at tick zero")
    outcomes = {"ROUTE_GOAL", "COLLISION", "OFFROAD", "CORRIDOR_VIOLATION",
                "MAP_COVERAGE_EXIT", "TASK_LOST", "RECORD_END", "TIME_LIMIT"}
    if (trace["outcome"] not in outcomes or frames[-1]["outcome"] != trace["outcome"]
            or any(f["outcome"] is not None for f in frames[:-1])):
        raise ValueError("incomplete, corrupted, or unsupported episode ending")
    steps = frames[1:]
    sources, events = Counter(), Counter()
    errors, returns = [], []
    rejected = stopped = 0
    first_brake = None
    for f in steps:
        if f["sample_valid"] is not True or type(f["accepted"]) is not bool:
            raise ValueError("invalid actual sample or proposal result")
        if f["proposal_tick"] != f["tick"] - 1:
            raise ValueError("proposal is not from this actual tick")
        source = f["source"]
        if source not in {"NEW", "REUSED", "BRAKE", "EMERGENCY"}:
            raise ValueError("unsupported execution source")
        xy = _xy(f["ego"]["xy"])
        speed, reward = float(f["ego"]["speed"]), float(f["reward"])
        if not math.isfinite(speed) or speed < 0 or not math.isfinite(reward):
            raise ValueError("nonfinite reward or invalid actual speed")
        selected = f["executed_xy"]
        if bool(selected) == (source == "EMERGENCY"):
            raise ValueError("selected trajectory availability differs from source")
        if selected:
            points = [_xy(p) for p in selected]
            dx, dy = xy[0] - points[0][0], xy[1] - points[0][1]
            errors.append(dict(tick=f["tick"], source=source,
                               position_m=math.sqrt(dx * dx + dy * dy)))
        if (not isinstance(f["events"], list) or
                any(not isinstance(x, str) for x in f["events"]) or
                len(set(f["events"])) != len(f["events"])):
            raise ValueError("invalid or duplicate actual event flags")
        sources[source] += 1
        events.update(f["events"])
        rejected += not f["accepted"]
        stopped += speed < 0.1
        returns.append(reward)
        if first_brake is None and source in ("BRAKE", "EMERGENCY"):
            first_brake = f["tick"]
    initial_speed = float(frames[0]["ego"]["speed"])
    if not math.isfinite(initial_speed) or initial_speed < 0:
        raise ValueError("invalid initial speed")
    squared = math.fsum(e["position_m"] ** 2 for e in errors)
    return dict(scene_id=trace["scene_id"], provider=trace["provider"],
                outcome=trace["outcome"], ticks=len(steps), initial_speed=initial_speed,
                events=dict(events), sources=dict(sources), rejected=rejected,
                braking=sources["BRAKE"] + sources["EMERGENCY"], stationary=stopped,
                first_braking_tick=first_brake, return_training=math.fsum(returns),
                tracking_samples=len(errors), tracking_position_squared=squared,
                tracking_position_rms=math.sqrt(squared / len(errors)) if errors else None,
                tracking_errors=errors)


def summarize(episodes):
    if not episodes or len({(e["scene_id"], e["provider"]) for e in episodes}) != len(episodes):
        raise ValueError("requires nonempty unique scene/provider episodes")
    ticks = sum(e["ticks"] for e in episodes)
    count = sum(e["tracking_samples"] for e in episodes)
    squared = math.fsum(e["tracking_position_squared"] for e in episodes)
    sources = sum((Counter(e["sources"]) for e in episodes), Counter())
    return dict(episodes=len(episodes), unique_scenes=len({e["scene_id"] for e in episodes}),
                ticks=ticks, mean_episode_ticks=ticks / len(episodes),
                outcomes=dict(Counter(e["outcome"] for e in episodes)),
                event_ticks=dict(sum((Counter(e["events"]) for e in episodes), Counter())),
                sources=dict(sources),
                success_rate=sum(e["outcome"] == "ROUTE_GOAL" for e in episodes) / len(episodes),
                collision_rate=sum("COLLISION" in e["events"] for e in episodes) / len(episodes),
                offroad_rate=sum("OFFROAD" in e["events"] for e in episodes) / len(episodes),
                raw_rejection_rate=sum(e["rejected"] for e in episodes) / ticks,
                braking_intervention_rate=sum(e["braking"] for e in episodes) / ticks,
                stationary_fraction=sum(e["stationary"] for e in episodes) / ticks,
                mean_return=math.fsum(e["return_training"] for e in episodes) / len(episodes),
                tracking_samples=count,
                tracking_position_rms=math.sqrt(squared / count) if count else None)
