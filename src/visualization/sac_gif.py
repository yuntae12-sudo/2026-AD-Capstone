"""Paired GIF from captured simulation evidence; rendering never calls a policy."""

from pathlib import Path
import textwrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from src.environment.womd_control.geometry import box
from .sac_rollout import frame_at, require_paired

COLORS = dict(
    ego="#2465ba",
    agent="#8e9bae",
    target="#bf8b22",
    actual="#138878",
    raw="#4e71df",
    reject="#d84e4b",
    chosen="#d28a19",
    brake="#c63c56",
)


def transform(points, origin, yaw):
    xy = np.asarray(points, float).reshape(-1, 2) - origin
    c, s = np.cos(yaw), np.sin(yaw)
    return xy @ np.array([[c, -s], [s, c]])


def selected_ticks(length, stride):
    if length < 1 or stride < 1:
        raise ValueError("positive length/stride required")
    return sorted(set(range(0, length, stride)) | {length - 1})


def draw_frame(evidence, tick):
    episodes = evidence["episodes"]
    initial = episodes[0]["frames"][0]["ego"]
    origin, yaw = np.asarray(initial["xy"]), initial["yaw"]
    current = [frame_at(ep, tick) for ep in episodes]
    centers = transform([f["ego"]["xy"] for f, _ in current], origin, yaw)
    center = centers.mean(axis=0)
    # Both panels always use the same scale and center, including diverged paths.
    width = max(70.0, float(np.ptp(centers[:, 0])) + 45.0)
    height = max(54.0, float(np.ptp(centers[:, 1])) + 30.0)
    width = max(width, height * 1.25)
    height = width / 1.25
    fig = plt.figure(figsize=(14.4, 9), dpi=100, facecolor="#f4f6f8")
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(5.4, 1.15, 0.55),
        left=0.055,
        right=0.975,
        top=0.85,
        bottom=0.10,
        hspace=0.38,
        wspace=0.18,
    )
    fig.text(
        0.055,
        0.96,
        "WAYMAX  /  CLOSED-LOOP PATH COMPARISON",
        weight="bold",
        fontsize=17,
        color="#253549",
    )
    fig.text(
        0.055,
        0.925,
        f"{episodes[0]['scene_id']}  |  {episodes[0]['provider'].removeprefix('fixed_').upper()}  |  t = {tick / 10:.1f} s",
        fontsize=11,
        color="#506070",
    )
    fig.text(
        0.975,
        0.925,
        "VISUALIZATION PREVIEW  /  not a performance verdict",
        ha="right",
        fontsize=9,
        color="#6e7884",
    )
    duration = max(len(ep["frames"]) - 1 for ep in episodes) / 10
    max_speed = max(f["ego"]["speed"] for ep in episodes for f in ep["frames"])
    for col, (ep, (frame, held)) in enumerate(zip(episodes, current)):
        ax = fig.add_subplot(grid[0, col], facecolor="#f9fafb")

        def xy(points):
            return transform(points, origin, yaw)

        def line(points, **kwargs):
            if len(points) > 1:
                pts = xy(points)
                ax.plot(pts[:, 0], pts[:, 1], **kwargs)

        ax.add_patch(
            Polygon(
                xy(ep["map"]["corridor"]),
                facecolor="#e3eee7",
                edgecolor="none",
                zorder=0,
            )
        )
        for feature in ep["map"]["lines"]:
            pts = xy(feature["xy"])
            low, high = pts.min(axis=0), pts.max(axis=0)
            if np.any(low > center + [width / 2 + 12, height / 2]) or np.any(
                high < center - [width / 2, height / 2]
            ):
                continue
            edge = feature["kind"] == "road_edge"
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="#9ea9b5" if edge else "#c8d0da",
                lw=1.2 if edge else 0.7,
                ls="--" if feature["kind"] == "lane" else "-",
                zorder=1,
            )
        line(ep["map"]["route"], color="#9bb8a6", lw=1.2, ls=":", zorder=2)
        history = ep["frames"][: min(tick + 1, len(ep["frames"]))]
        line([f["ego"]["xy"] for f in history], color=COLORS["actual"], lw=3, zorder=4)
        line(
            frame["raw_xy"],
            color=COLORS["raw"] if frame["accepted"] else COLORS["reject"],
            lw=2,
            ls="--",
            zorder=5,
        )
        line(frame["executed_xy"], color=COLORS["chosen"], lw=1.7, zorder=6)
        for v in frame["objects"] + [frame["ego"]]:
            is_ego = v is frame["ego"]
            color = (
                COLORS["ego"]
                if is_ego
                else (
                    COLORS["target"]
                    if v["id"] == frame["target_id"]
                    else COLORS["agent"]
                )
            )
            vertices = xy(box(np.asarray(v["xy"]), v["yaw"], v["length"], v["width"]))
            collision = (
                (v["id"] in frame["colliding_ids"])
                if not is_ego
                else bool(frame["colliding_ids"])
            )
            ax.add_patch(
                Polygon(
                    vertices,
                    facecolor=color,
                    edgecolor=COLORS["reject"] if collision else "#ffffff",
                    lw=2 if collision else 0.6,
                    zorder=8 if is_ego else 7,
                )
            )
            if is_ego:
                p = xy([v["xy"]])[0]
                heading = v["yaw"] - yaw
                ax.arrow(
                    *p,
                    2 * np.cos(heading),
                    2 * np.sin(heading),
                    width=0.15,
                    color="white",
                    zorder=9,
                )
        ax.set_xlim(center[0] - width / 2 + 10, center[0] + width / 2 + 10)
        ax.set_ylim(center[1] - height / 2, center[1] + height / 2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
        ax.set_xlabel("Forward from initial heading [m]", fontsize=9)
        if col == 0:
            ax.set_ylabel("Left from initial heading [m]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(ep["label"], loc="left", fontsize=12, fontweight="bold", pad=11)
        outcome = frame["outcome"] or (
            ep["outcome"] if tick >= len(ep["frames"]) - 1 else None
        )
        badge = f"{frame['source']}  |  {frame['ego']['speed']:.2f} m/s"
        if outcome:
            badge += f"  |  {outcome}" + (" [held]" if held else "")
        reason = "\n".join(
            textwrap.wrap(
                frame["reason"]
                + (
                    ": " + ", ".join(frame["violations"]) if frame["violations"] else ""
                ),
                55,
            )
        )
        ax.text(
            0.025,
            0.975,
            badge + "\n" + reason,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#ad3545" if frame["source"] in ("BRAKE", "EMERGENCY") else "#253549",
            bbox=dict(
                boxstyle="round,pad=.5",
                facecolor="white",
                alpha=0.94,
                edgecolor="#dde3e9",
            ),
            zorder=12,
        )
        metrics = fig.add_subplot(grid[1, col])
        times = [f["time_s"] for f in history]
        metrics.plot(
            times, [f["ego"]["speed"] for f in history], color=COLORS["ego"], lw=1.8
        )
        for f in history[1:]:
            if f["source"] in ("BRAKE", "EMERGENCY"):
                metrics.axvspan(
                    f["time_s"] - 0.1, f["time_s"], color=COLORS["brake"], alpha=0.13
                )
        metrics.axvline(tick / 10, color="#7c8794", lw=0.8)
        metrics.set_xlim(0, max(duration, 0.1))
        metrics.set_ylim(0, max(max_speed * 1.15, 1))
        metrics.set_ylabel("m/s", fontsize=8)
        metrics.grid(alpha=0.15)
        metrics.tick_params(labelsize=8)
        strip = fig.add_subplot(grid[2, col])
        for f in history[1:]:
            strip.broken_barh(
                [(f["time_s"] - 0.1, 0.1)],
                (0, 1),
                facecolors=COLORS["actual"] if f["accepted"] else COLORS["reject"],
            )
        strip.set_xlim(0, max(duration, 0.1))
        strip.set_ylim(0, 1)
        strip.set_yticks([])
        strip.set_xlabel("Simulation time [s]  /  proposal acceptance", fontsize=8)
        strip.tick_params(labelsize=8)
    handles = [
        Line2D([0], [0], color=COLORS[c], lw=2, ls=ls, label=label)
        for c, ls, label in (
            ("actual", "-", "Actual ego trail"),
            ("raw", "--", "Accepted proposal"),
            ("reject", "--", "Rejected proposal"),
            ("chosen", "-", "Selected control path"),
            ("brake", "-", "Braking (speed-chart shading)"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.032),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.017,
        "Vehicles: blue = ego, gray = logged actors, gold = FOLLOW target. Proposals are from the preceding control tick.",
        ha="center",
        fontsize=8,
        color="#647182",
    )
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return image


def render(evidence, output, stride=1):
    require_paired(evidence["episodes"])
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    length = max(len(ep["frames"]) for ep in evidence["episodes"])
    ticks = selected_ticks(length, stride)
    frames = []
    for tick in ticks:
        frame = draw_frame(evidence, tick)
        if tick == ticks[-1]:
            frame.save(output.with_suffix(".png"))
        frames.append(frame.quantize(colors=128))
    # Physical time remains 1x even when rendering every Nth simulation tick.
    durations = [(b - a) * 100 for a, b in zip(ticks, ticks[1:])] + [1500]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return dict(
        frames=len(frames),
        simulation_duration_s=(length - 1) * 0.1,
        playback="1x plus 1.5 s final hold",
        dimensions=frames[0].size,
    )
