#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "nbody_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"step", "time", "particle", "x", "y", "z", "mass"}
MAX_OUTPUT_FRAMES = 160
FIGURE_SIZE = (8, 7)
FIGURE_DPI = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a fixed-scale 3D GIF from an N-body trajectory CSV."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--trail-length",
        type=int,
        default=20,
        help="Number of animated frames retained in each trail; 0 disables trails.",
    )
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--show-center-of-mass", action="store_true")
    parser.add_argument(
        "--rotate-camera",
        action="store_true",
        help="Rotate the azimuth once over the animation.",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=64,
        help="Maximum number of particles displayed, selected by particle id.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.trail_length < 0:
        parser.error("--trail-length must be non-negative")
    if args.max_particles <= 0:
        parser.error("--max-particles must be positive")
    if not math.isfinite(args.elev) or not math.isfinite(args.azim):
        parser.error("camera angles must be finite")
    if args.dt is not None and (not math.isfinite(args.dt) or args.dt <= 0):
        parser.error("--dt must be finite and positive")
    if args.epsilon is not None and (not math.isfinite(args.epsilon) or args.epsilon < 0):
        parser.error("--epsilon must be finite and non-negative")
    return args


def resolve_path(path):
    return path if path.is_absolute() else ROOT / path


def read_trajectory(path):
    if not path.is_file():
        raise RuntimeError(f"trajectory CSV does not exist: {path}")

    frames = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("trajectory CSV has no header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "trajectory CSV is missing columns: " + ", ".join(sorted(missing))
            )

        previous_step = -1
        for line_number, row in enumerate(reader, start=2):
            try:
                step = int(row["step"])
                time = float(row["time"])
                particle = int(row["particle"])
                position = tuple(float(row[axis]) for axis in ("x", "y", "z"))
                mass = float(row["mass"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"invalid numeric value at CSV line {line_number}") from error

            if step < 0 or particle < 0:
                raise RuntimeError(f"negative step or particle id at CSV line {line_number}")
            if (not all(math.isfinite(value) for value in (time, *position, mass)) or
                    mass <= 0.0):
                raise RuntimeError(f"non-finite value at CSV line {line_number}")
            if step < previous_step:
                raise RuntimeError(f"steps are not monotonic at CSV line {line_number}")

            frame = frames.setdefault(step, {"time": time, "positions": {}})
            if not math.isclose(frame["time"], time, rel_tol=0.0, abs_tol=1.0e-15):
                raise RuntimeError(f"inconsistent time within step {step}")
            if particle in frame["positions"]:
                raise RuntimeError(f"duplicate particle {particle} at step {step}")
            frame["positions"][particle] = position
            frame.setdefault("masses", {})[particle] = mass
            previous_step = step

    if not frames:
        raise RuntimeError("trajectory CSV contains no data rows")

    steps = sorted(frames)
    particle_ids = sorted(frames[steps[0]]["positions"])
    if not particle_ids:
        raise RuntimeError("first trajectory frame contains no particles")
    expected_particles = set(particle_ids)
    initial_masses = None
    previous_time = -math.inf
    for step in steps:
        actual_particles = set(frames[step]["positions"])
        if actual_particles != expected_particles:
            raise RuntimeError(
                f"step {step} has {len(actual_particles)} particles; "
                f"expected the same {len(expected_particles)} particle ids in every frame"
            )
        if set(frames[step]["masses"]) != expected_particles:
            raise RuntimeError(f"step {step} has incomplete mass data")
        masses = tuple(frames[step]["masses"][particle] for particle in particle_ids)
        if initial_masses is None:
            initial_masses = masses
        elif masses != initial_masses:
            raise RuntimeError(f"particle masses changed at step {step}")
        if frames[step]["time"] <= previous_time:
            raise RuntimeError(f"times are not strictly increasing at step {step}")
        previous_time = frames[step]["time"]

    return steps, particle_ids, frames


def select_animation_steps(steps, frame_stride):
    selected = steps[::frame_stride]
    if selected[-1] != steps[-1]:
        selected.append(steps[-1])

    if len(selected) > MAX_OUTPUT_FRAMES:
        indices = np.linspace(0, len(selected) - 1, MAX_OUTPUT_FRAMES, dtype=int)
        selected = [selected[index] for index in indices]
    return selected


def fixed_axis_limits(steps, particle_ids, frames):
    positions = np.asarray(
        [frames[step]["positions"][particle] for step in steps for particle in particle_ids],
        dtype=float,
    )
    minima = positions.min(axis=0)
    maxima = positions.max(axis=0)
    center = 0.5 * (minima + maxima)
    span = float(np.max(maxima - minima))
    if span == 0.0:
        span = 1.0
    half_span = 0.55 * span
    return tuple((center[index] - half_span, center[index] + half_span) for index in range(3))


def motion_metrics(steps, particle_ids, frames, spatial_span):
    initial_step = steps[0]
    final_step = steps[-1]
    displacements = [
        math.dist(
            frames[initial_step]["positions"][particle],
            frames[final_step]["positions"][particle],
        )
        for particle in particle_ids
    ]
    rms_displacement = math.sqrt(
        sum(displacement * displacement for displacement in displacements)
        / len(displacements)
    )
    maximum_displacement = max(displacements)
    return {
        "rms_displacement": rms_displacement,
        "maximum_displacement": maximum_displacement,
        "spatial_span": spatial_span,
        "relative_rms_motion": rms_displacement / spatial_span,
        "relative_maximum_motion": maximum_displacement / spatial_span,
    }


def make_gif(all_steps, steps, particle_ids, frames, output, args):
    displayed_particles = particle_ids[: args.max_particles]
    limits = fixed_axis_limits(all_steps, displayed_particles, frames)
    colors = plt.get_cmap("hsv")(np.linspace(0.0, 1.0, len(displayed_particles), endpoint=False))
    masses = np.asarray([frames[all_steps[0]]["masses"][particle]
                         for particle in displayed_particles])
    marker_sizes = 14.0 + 30.0 * np.sqrt(masses / masses.max())

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=args.elev, azim=args.azim)

    initial_positions = np.asarray(
        [frames[steps[0]]["positions"][particle] for particle in displayed_particles]
    )
    scatter = ax.scatter(
        initial_positions[:, 0],
        initial_positions[:, 1],
        initial_positions[:, 2],
        s=marker_sizes,
        c=colors,
        depthshade=True,
    )
    trails = []
    if args.trail_length > 0:
        trails = [ax.plot([], [], [], color=color, linewidth=0.8, alpha=0.55)[0] for color in colors]
    com_marker = None
    if args.show_center_of_mass:
        com_marker = ax.scatter([], [], [], marker="x", s=55, color="black",
                                linewidths=1.6, label="mass-weighted center of mass")
        ax.scatter([], [], [], s=30, color="gray",
                   label=r"marker size $\propto \sqrt{m/m_{max}}$")
        ax.legend(loc="upper right", fontsize=8)

    def update(frame_index):
        step = steps[frame_index]
        positions = np.asarray(
            [frames[step]["positions"][particle] for particle in displayed_particles]
        )
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        if com_marker is not None:
            total_mass = float(masses.sum())
            com = np.sum(positions * masses[:, None], axis=0) / total_mass
            com_marker._offsets3d = ([com[0]], [com[1]], [com[2]])

        if trails:
            first = max(0, frame_index - args.trail_length + 1)
            trail_steps = steps[first : frame_index + 1]
            for particle, line in zip(displayed_particles, trails):
                trail_positions = np.asarray(
                    [frames[trail_step]["positions"][particle] for trail_step in trail_steps]
                )
                line.set_data(trail_positions[:, 0], trail_positions[:, 1])
                line.set_3d_properties(trail_positions[:, 2])

        if args.rotate_camera and len(steps) > 1:
            ax.view_init(
                elev=args.elev,
                azim=args.azim + 360.0 * frame_index / (len(steps) - 1),
            )
        details = []
        if args.epsilon is not None:
            details.append(f"epsilon={args.epsilon:g}")
        if args.dt is not None:
            details.append(f"dt={args.dt:g}")
        suffix = ", " + ", ".join(details) if details else ""
        ax.set_title(f"N={len(particle_ids)}, t={frames[step]['time']:.4f}{suffix}\n"
                     f"step={step}, showing {len(displayed_particles)} particles")
        return (scatter, *trails, *((com_marker,) if com_marker is not None else ()))

    animation = FuncAnimation(
        fig,
        update,
        frames=len(steps),
        interval=1000.0 / args.fps,
        blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=args.fps), dpi=FIGURE_DPI)
    plt.close(fig)
    metrics = motion_metrics(
        steps,
        displayed_particles,
        frames,
        limits[0][1] - limits[0][0],
    )
    return displayed_particles, limits, metrics


def main():
    args = parse_args()
    trajectory = resolve_path(args.trajectory)
    output = resolve_path(args.output)

    try:
        all_steps, particle_ids, frames = read_trajectory(trajectory)
        animation_steps = select_animation_steps(all_steps, args.frame_stride)
        displayed_particles, limits, metrics = make_gif(
            all_steps, animation_steps, particle_ids, frames, output, args
        )
    except ImportError:
        print("Pillow is required to write GIF files. Install pillow, then retry.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"trajectory,{trajectory}")
    print(f"output,{output}")
    print(f"input_frames,{len(all_steps)}")
    print(f"output_frames,{len(animation_steps)}")
    print(f"first_step,{animation_steps[0]}")
    print(f"last_step,{animation_steps[-1]}")
    print("selected_steps," + ",".join(map(str, animation_steps)))
    print(f"particles_per_frame,{len(particle_ids)}")
    print(f"displayed_particles,{len(displayed_particles)}")
    print(f"resolution,{FIGURE_SIZE[0] * FIGURE_DPI:.0f}x{FIGURE_SIZE[1] * FIGURE_DPI:.0f}")
    print(f"fixed_xlim,{limits[0][0]:.17g},{limits[0][1]:.17g}")
    print(f"fixed_ylim,{limits[1][0]:.17g},{limits[1][1]:.17g}")
    print(f"fixed_zlim,{limits[2][0]:.17g},{limits[2][1]:.17g}")
    print(f"initial_final_rms_displacement,{metrics['rms_displacement']:.17g}")
    print(f"maximum_displacement,{metrics['maximum_displacement']:.17g}")
    print(f"spatial_span,{metrics['spatial_span']:.17g}")
    print(f"relative_rms_motion,{metrics['relative_rms_motion']:.17g}")
    print(f"relative_maximum_motion,{metrics['relative_maximum_motion']:.17g}")
    if metrics["relative_rms_motion"] < 0.02:
        print(
            "warning: relative RMS motion is below 0.02; particle motion may be "
            "difficult to see at the fixed spatial scale",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
