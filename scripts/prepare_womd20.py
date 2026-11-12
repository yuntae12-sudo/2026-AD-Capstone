#!/usr/bin/env python3
"""Index, inspect, and replay 20-second WOMD without a trained policy.

Run from the repository with the its-merge environment. Outputs default to E:.
This command defaults to CPU and does not alter the existing training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scenarios.womd20 import (  # noqa: E402
    DEFAULT_OUTPUT, DEFAULT_ROOT, parse_scenario, raw_records, scenario_to_waymax,
)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def replay(scenario, state, output, *, render_stride=5, render=True):
    """Advance through every frame, compare replay values, and optionally render."""
    import jax
    import numpy as np
    from waymax import datatypes

    times = np.asarray(scenario.timestamps_seconds)
    relative_times = times - times[0]
    total = len(times)
    advance = jax.jit(lambda s: datatypes.update_state_by_log(s, num_steps=1))
    current = jax.jit(lambda s: s.current_sim_trajectory)
    sdc = scenario.sdc_track_index
    images, image_indices, rows = [], [], []
    if render:
        from PIL import Image, ImageDraw
        from waymax import visualization
    for t in range(total):
        if int(state.timestep) != t:
            raise AssertionError("Replay timestep mismatch")
        if bool(state.is_done) != (t == total - 1):
            raise AssertionError("Unexpected termination boundary")
        if int(state.remaining_timesteps) != total - t - 1:
            raise AssertionError("Unexpected remaining timesteps")
        simulated = jax.device_get(current(state))
        valid = np.asarray(simulated.valid)[:, 0]
        expected_valid = np.array([track.states[t].valid for track in scenario.tracks])
        np.testing.assert_array_equal(valid, expected_valid)
        for name, proto_name in (("x", "center_x"), ("y", "center_y"), ("yaw", "heading")):
            expected = np.asarray([getattr(track.states[t], proto_name) for track in scenario.tracks], np.float32)
            np.testing.assert_array_equal(np.asarray(getattr(simulated, name))[:, 0][valid], expected[valid])
        lights = state.log_traffic_light
        light_valid = np.asarray(lights.valid)[:, t]
        observed_lights = {int(np.asarray(lights.lane_ids)[i, t]): int(np.asarray(lights.state)[i, t])
                           for i in np.flatnonzero(light_valid)}
        expected_lights = ({int(light.lane): int(light.state)
                            for light in scenario.dynamic_map_states[t].lane_states}
                           if scenario.dynamic_map_states else {})
        if observed_lights != expected_lights:
            raise AssertionError(f"Traffic-light mismatch at frame {t}")
        rows.append(dict(frame=t, elapsed_s=float(relative_times[t]),
                         valid_objects=int(valid.sum()), valid_traffic_lights=int(light_valid.sum()),
                         sdc_valid=bool(valid[sdc]), sdc_x=float(simulated.x[sdc, 0]),
                         sdc_y=float(simulated.y[sdc, 0]), done=bool(state.is_done)))
        if render and (t % render_stride == 0 or t == total - 1):
            image = Image.fromarray(visualization.plot_simulator_state(
                state, use_log_traj=False,
                viz_config={"show_agent_id": False, "px_per_meter": 3.0},
            ))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 24), fill="white")
            draw.text((5, 5), f"LOG REPLAY  frame {t}/{total-1}  t={relative_times[t]:.2f}s  SDC valid={bool(valid[sdc])}", fill="black")
            images.append(image)
            image_indices.append(t)
        if t < total - 1:
            state = advance(state)
    # All replayed fields, including dimensions, timestamps and masks, must match.
    for simulated, logged in zip(jax.tree_util.tree_leaves(state.sim_trajectory),
                                 jax.tree_util.tree_leaves(state.log_trajectory)):
        np.testing.assert_array_equal(np.asarray(simulated), np.asarray(logged))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "replay_frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if images:
        durations = [max(10, int(round((relative_times[b] - relative_times[a]) * 1000 / 10)) * 10)
                     for a, b in zip(image_indices[:-1], image_indices[1:])]
        # Last frame stays visible for one source sample, rounded to GIF's 10 ms.
        durations.append(max(10, int(round(float(np.median(np.diff(times))) * 100)) * 10))
        images[0].save(output / "replay.gif", save_all=True, append_images=images[1:],
                       duration=durations, loop=0)
        images[-1].save(output / "last_frame.png")
    return dict(checked_frames=total, frames_after_9s=int((relative_times > 9).sum()),
                final_frame=int(state.timestep), final_done=bool(state.is_done),
                remaining_timesteps=int(state.remaining_timesteps),
                all_logged_fields_match=True, traffic_light_states_match=True,
                mode="Waymax update_state_by_log (not controlled dynamics / policy evaluation)",
                rendered_frames=len(images), render_stride=render_stride)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("index", "inspect", "replay"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / command)
        if command == "index":
            p.add_argument("--limit-shards", type=positive_int,
                           help="Pilot scan of the first N shards; omit for all 1000. Resumes automatically.")
        else:
            p.add_argument("--shard-index", type=int, default=0, choices=range(1000), metavar="0..999")
            p.add_argument("--record-indices", nargs="+", type=int, default=[0, 1, 2])
            if command == "replay":
                p.add_argument("--render-stride", type=positive_int, default=5,
                               help="Render every Nth frame; ALL source frames are checked.")
                p.add_argument("--no-gif", action="store_true", help="Only numeric replay verification")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"Source directory not found: {args.root}")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    if args.command == "index":
        if args.limit_shards is not None and args.limit_shards > 1000:
            parser.error("--limit-shards must be <= 1000")
        from src.scenarios.womd20_index import scan_index
        result = scan_index(args.root, args.output, args.limit_shards)
        return int(bool(result["failed_shards"] or result["duplicate_scenario_ids"]))
    if any(i < 0 for i in args.record_indices):
        parser.error("--record-indices must be nonnegative")
    path = args.root / f"training_20s.tfrecord-{args.shard_index:05d}-of-01000"
    if not path.is_file():
        parser.error(f"Source shard not found: {path}")
    wanted = set(args.record_indices)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, raw in enumerate(raw_records(path)):
        if index not in wanted:
            continue
        wanted.remove(index)
        report = {"source_shard": path.name, "record_index": index}
        try:
            scenario = parse_scenario(raw)
            state, conversion = scenario_to_waymax(scenario)
            report.update(conversion, status="ok")
            if args.command == "replay":
                destination = args.output / f"shard_{args.shard_index:05d}_record_{index:05d}"
                report["replay"] = replay(scenario, state, destination,
                                          render_stride=args.render_stride, render=not args.no_gif)
                report["output"] = str(destination)
        except Exception as exc:
            report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        reports.append(report)
        write_json(args.output / "report.json", reports)
        print(json.dumps(report), flush=True)
        if not wanted:
            break
    for index in sorted(wanted):
        reports.append(dict(source_shard=path.name, record_index=index,
                            status="error", error="Record not found"))
    write_json(args.output / "report.json", reports)
    print(f"Report: {args.output / 'report.json'}", flush=True)
    return int(any(report["status"] != "ok" for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
