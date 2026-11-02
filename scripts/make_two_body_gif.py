#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "nbody_matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "figures" / "two_body_verlet.gif"
REQUIRED_COLUMNS = {"step", "time", "particle", "x", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a two-body orbit GIF.")
    parser.add_argument(
        "--input", "--trajectory",
        dest="trajectory",
        type=Path,
        required=True,
        help="Input two-body trajectory CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output GIF path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Animation frames per second.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--trail-length", type=int, default=-1,
        help="Selected frames retained in trails; -1 means complete, 0 disables trails.",
    )
    parser.add_argument("--period", type=float)
    parser.add_argument("--show-center-of-mass", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or args.frame_stride <= 0:
        parser.error("--fps and --frame-stride must be positive")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if args.trail_length < -1:
        parser.error("--trail-length must be -1, 0, or positive")
    if args.period is not None and (not math.isfinite(args.period) or args.period <= 0):
        parser.error("--period must be finite and positive")
    return args


def resolve_path(path):
    if path.is_absolute():
        return path
    return ROOT / path


def require_trajectory(path):
    if path.exists():
        return

    print(f"Missing trajectory CSV: {path}", file=sys.stderr)
    print("Generate it with:", file=sys.stderr)
    print(
        "./bin/nbody_serial --N 2 --steps 628 --dt 0.01 --epsilon 0.01 "
        "--init two-body --integrator verlet --save-every 5 "
        "--output results/two_body_verlet.csv "
        "--energy-output results/two_body_verlet_energy.csv",
        file=sys.stderr,
    )
    sys.exit(1)


def read_trajectory(path):
    frames = {}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError("trajectory CSV is missing columns: " +
                               ", ".join(sorted(missing)))
        has_mass = "mass" in (reader.fieldnames or ())
        previous_step = -1
        for line_number, row in enumerate(reader, start=2):
            try:
                step, particle = int(row["step"]), int(row["particle"])
                values = [float(row[key]) for key in ("time", "x", "y")]
                mass = float(row["mass"]) if has_mass else None
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"invalid numeric value at CSV line {line_number}") from error
            time, x, y = values
            if step < previous_step:
                raise RuntimeError(f"steps are not monotonic at CSV line {line_number}")
            if (not all(math.isfinite(value) for value in values) or
                    (mass is not None and (not math.isfinite(mass) or mass <= 0))):
                raise RuntimeError(f"non-finite value or non-positive mass at line {line_number}")
            frame = frames.setdefault(step, {"time": time, "particles": {}})
            if not math.isclose(frame["time"], time, rel_tol=0, abs_tol=1e-15):
                raise RuntimeError(f"inconsistent time within step {step}")
            if particle in frame["particles"]:
                raise RuntimeError(f"duplicate particle {particle} at step {step}")
            frame["particles"][particle] = {"x": x, "y": y, "mass": mass}
            previous_step = step

    steps = list(frames)
    if not steps or steps[0] < 0:
        raise RuntimeError("trajectory contains no valid frames")
    expected_ids = {0, 1}
    initial_masses = None
    previous_time = -math.inf
    for step in steps:
        frame = frames[step]
        if set(frame["particles"]) != expected_ids:
            raise RuntimeError(f"step {step} does not contain exactly particle IDs 0 and 1")
        masses = tuple(frame["particles"][particle]["mass"] for particle in (0, 1))
        if has_mass:
            if initial_masses is None:
                initial_masses = masses
            elif masses != initial_masses:
                raise RuntimeError(f"particle mass changed at step {step}")
        if frame["time"] <= previous_time:
            raise RuntimeError(f"times are not strictly increasing at step {step}")
        previous_time = frame["time"]
    return steps, frames


def select_steps(steps, frame_stride, max_frames):
    selected = steps[::frame_stride]
    if selected[-1] != steps[-1]:
        selected.append(steps[-1])
    if max_frames is not None and len(selected) > max_frames:
        indices = np.linspace(0, len(selected) - 1, max_frames, dtype=int)
        selected = [selected[index] for index in indices]
    if selected[0] != steps[0] or selected[-1] != steps[-1]:
        raise RuntimeError("frame selection did not preserve both endpoints")
    return selected


def axis_limits(steps, frames):
    xs = []
    ys = []

    for step in steps:
        for particle in frames[step]["particles"].values():
            xs.append(particle["x"])
            ys.append(particle["y"])

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min)
    if span == 0.0:
        span = 1.0

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_span = 0.55 * span

    return (
        x_center - half_span,
        x_center + half_span,
        y_center - half_span,
        y_center + half_span,
    )


def make_gif(steps, frames, output, args):
    output.parent.mkdir(parents=True, exist_ok=True)
    x_min, x_max, y_min, y_max = axis_limits(steps, frames)

    fig, ax = plt.subplots()
    colors = ("tab:blue", "tab:orange")
    trail0, = ax.plot([], [], linewidth=1.3, color=colors[0], label="particle 0")
    trail1, = ax.plot([], [], linewidth=1.3, color=colors[1], label="particle 1")
    point0, = ax.plot([], [], marker="o", markersize=6, color=colors[0])
    point1, = ax.plot([], [], marker="o", markersize=6, color=colors[1])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x coordinate (arbitrary units)")
    ax.set_ylabel("y coordinate (arbitrary units)")
    ax.grid(True, alpha=0.3)
    if args.show_center_of_mass:
        initial = frames[steps[0]]["particles"]
        if any(initial[p]["mass"] is None for p in (0, 1)):
            raise RuntimeError("--show-center-of-mass requires a trajectory mass column")
        total_mass = initial[0]["mass"] + initial[1]["mass"]
        com_x = sum(initial[p]["mass"] * initial[p]["x"] for p in (0, 1)) / total_mass
        com_y = sum(initial[p]["mass"] * initial[p]["y"] for p in (0, 1)) / total_mass
        ax.plot(com_x, com_y, marker="+", color="black", markersize=9,
                markeredgewidth=1.5, linestyle="none", label="center of mass")
    ax.legend(loc="upper right", fontsize=8)

    xs0 = []
    ys0 = []
    xs1 = []
    ys1 = []

    def update(frame_index):
        step = steps[frame_index]
        time = frames[step]["time"]
        x0 = frames[step]["particles"][0]["x"]
        y0 = frames[step]["particles"][0]["y"]
        x1 = frames[step]["particles"][1]["x"]
        y1 = frames[step]["particles"][1]["y"]

        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)

        first = 0 if args.trail_length == -1 else max(0, len(xs0) - args.trail_length)
        if args.trail_length == 0:
            trail0.set_data([], []); trail1.set_data([], [])
        else:
            trail0.set_data(xs0[first:], ys0[first:])
            trail1.set_data(xs1[first:], ys1[first:])
        point0.set_data([x0], [y0])
        point1.set_data([x1], [y1])
        phase = f", t/P={time / args.period:.3f}" if args.period else ""
        ax.set_title(f"Elliptic two-body orbit: step={step}, t={time:.4f}{phase}")

        return trail0, trail1, point0, point1

    animation = FuncAnimation(fig, update, frames=len(steps),
                              interval=1000 / args.fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)


def main():
    args = parse_args()
    trajectory = resolve_path(args.trajectory)
    output = resolve_path(args.output)

    try:
        require_trajectory(trajectory)
        all_steps, frames = read_trajectory(trajectory)
        steps = select_steps(all_steps, args.frame_stride, args.max_frames)
        make_gif(steps, frames, output, args)
    except ImportError:
        print("Pillow is required to write GIF files. Install pillow, then retry.", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"input,{trajectory}")
    print(f"output,{output}")
    print(f"input_frames,{len(all_steps)}")
    print(f"selected_frames,{len(steps)}")
    print(f"first_step,{steps[0]}")
    print(f"last_step,{steps[-1]}")
    print("selected_steps," + ",".join(map(str, steps)))


if __name__ == "__main__":
    main()
