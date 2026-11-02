#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "nbody_matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "nbody_serial"
RESULTS_ROOT = ROOT / "results" / "validation" / "dt_study_v4"
FIGURES_DIR = ROOT / "figures"

G = 1.0
EPSILON = 0.01
M1 = 1.0
M2 = 1.0e-3
SEPARATION = 1.0
INTEGRATORS = ("euler", "verlet")
ROUND_OFF_FLOOR = 100.0 * sys.float_info.epsilon
TIME_TOLERANCE_FACTOR = 64.0
TRAJECTORY_FIELDS = (
    "step", "time", "particle", "x", "y", "z", "vx", "vy", "vz", "mass"
)
ENERGY_FIELDS = ("step", "time", "kinetic", "potential", "total", "relative_error")
PROFILES = {
    "smoke": {"periods": 1, "steps_per_period": (250, 500)},
    "final": {
        "periods": 5,
        "steps_per_period": (250, 500, 1000, 2000, 4000, 8000),
    },
}

SUMMARY_FIELDS = [
    "profile", "initializer", "integrator", "G", "m1", "m2",
    "initial_separation", "periods", "steps_per_period", "steps", "dt",
    "theoretical_period", "simulated_time", "epsilon", "save_every",
    "trajectory_frames", "energy_samples", "position_error_window_periods",
    "position_error_window_frames", "rms_includes_initial",
    "initial_energy", "max_relative_position_error",
    "rms_relative_position_error", "final_relative_position_error",
    "full_duration_max_relative_position_error",
    "full_duration_rms_relative_position_error",
    "full_duration_final_relative_position_error",
    "max_relative_energy_error", "rms_relative_energy_error",
    "final_relative_energy_error", "max_relative_separation_error",
    "rms_relative_separation_error", "final_relative_separation_error",
    "order_error_metric", "order_roundoff_floor",
    "observed_position_order_max_from_previous",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Study timestep convergence for the softened barycentric orbit."
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    return parser.parse_args()


def require_binary():
    if not BIN.exists():
        raise RuntimeError(f"missing executable: {BIN}; run `make` first")


def theoretical_omega():
    softened = (SEPARATION * SEPARATION + EPSILON * EPSILON) ** 1.5
    return math.sqrt(G * (M1 + M2) / softened)


def expected_saved_steps(steps, save_every):
    expected = list(range(0, steps + 1, save_every))
    if expected[-1] != steps:
        expected.append(steps)
    return expected


def parse_integer(value, label, path):
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise RuntimeError(f"non-integer {label} in {path}: {value}")
    return int(parsed)


def validate_time(actual, step, dt, path):
    expected = step * dt
    tolerance = TIME_TOLERANCE_FACTOR * sys.float_info.epsilon * max(1.0, abs(expected))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            f"time mismatch at step {step} in {path}: {actual} != {expected} "
            f"within {tolerance}"
        )


def read_trajectory(path, particle_count, steps, save_every, dt):
    frames = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRAJECTORY_FIELDS:
            raise RuntimeError(f"unexpected trajectory schema in {path}: {reader.fieldnames}")
        for raw in reader:
            row = {key: float(value) for key, value in raw.items()}
            if not all(math.isfinite(value) for value in row.values()):
                raise RuntimeError(f"non-finite trajectory row in {path}: {raw}")
            step = parse_integer(raw["step"], "step", path)
            particle = parse_integer(raw["particle"], "particle ID", path)
            validate_time(row["time"], step, dt, path)
            frame = frames.setdefault(step, {})
            if particle in frame:
                raise RuntimeError(f"duplicate particle {particle} at step {step} in {path}")
            frame[particle] = row

    expected_steps = expected_saved_steps(steps, save_every)
    if sorted(frames) != expected_steps:
        raise RuntimeError(f"unexpected trajectory steps in {path}: {sorted(frames)}")
    expected_ids = set(range(particle_count))
    initial_masses = None
    for step in expected_steps:
        frame = frames[step]
        if set(frame) != expected_ids:
            raise RuntimeError(f"incomplete particle IDs at step {step} in {path}")
        masses = [frame[particle]["mass"] for particle in range(particle_count)]
        if initial_masses is None:
            initial_masses = masses
        elif masses != initial_masses:
            raise RuntimeError(f"particle mass changed at step {step} in {path}")
    if sum(len(frame) for frame in frames.values()) != len(expected_steps) * particle_count:
        raise RuntimeError(f"unexpected trajectory row count in {path}")
    return [frames[step] for step in expected_steps]


def read_energy(path, steps, save_every, dt):
    rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ENERGY_FIELDS:
            raise RuntimeError(f"unexpected energy schema in {path}: {reader.fieldnames}")
        for raw in reader:
            row = {key: float(value) for key, value in raw.items()}
            if not all(math.isfinite(value) for value in row.values()):
                raise RuntimeError(f"non-finite energy row in {path}: {raw}")
            step = parse_integer(raw["step"], "step", path)
            validate_time(row["time"], step, dt, path)
            if step in rows:
                raise RuntimeError(f"duplicate energy row at step {step} in {path}")
            rows[step] = row

    expected_steps = expected_saved_steps(steps, save_every)
    if sorted(rows) != expected_steps:
        raise RuntimeError(f"unexpected energy steps in {path}: {sorted(rows)}")
    return [rows[step] for step in expected_steps]


def validate_matching_outputs(frames, energy_rows, path):
    if len(frames) != len(energy_rows):
        raise RuntimeError(f"trajectory/energy frame count mismatch for {path}")
    for frame, energy in zip(frames, energy_rows):
        step = int(energy["step"])
        if any(int(row["step"]) != step for row in frame.values()):
            raise RuntimeError(f"trajectory/energy step mismatch at {path}")
        if any(row["time"] != energy["time"] for row in frame.values()):
            raise RuntimeError(f"trajectory/energy time mismatch at step {step} in {path}")


def rms_without_initial(values):
    samples = values[1:]
    if not samples:
        raise RuntimeError("RMS requires at least one sample after the initial state")
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def position_and_separation_errors(frames, omega):
    initial = frames[0]
    masses = [initial[particle]["mass"] for particle in (0, 1)]
    total_mass = sum(masses)
    relative_position = [
        initial[1][axis] - initial[0][axis] for axis in ("x", "y", "z")
    ]
    relative_velocity = [
        initial[1][axis] - initial[0][axis] for axis in ("vx", "vy", "vz")
    ]
    center_of_mass = [
        sum(masses[p] * initial[p][axis] for p in (0, 1)) / total_mass
        for axis in ("x", "y", "z")
    ]
    center_of_mass_velocity = [
        sum(masses[p] * initial[p][axis] for p in (0, 1)) / total_mass
        for axis in ("vx", "vy", "vz")
    ]

    position_errors = []
    separation_errors = []
    for frame in frames:
        time = frame[0]["time"]
        cosine = math.cos(omega * time)
        sine = math.sin(omega * time)
        exact_relative = [
            relative_position[axis] * cosine
            + relative_velocity[axis] * sine / omega
            for axis in range(3)
        ]
        exact_com = [
            center_of_mass[axis] + center_of_mass_velocity[axis] * time
            for axis in range(3)
        ]
        exact_positions = (
            [
                exact_com[axis] - masses[1] * exact_relative[axis] / total_mass
                for axis in range(3)
            ],
            [
                exact_com[axis] + masses[0] * exact_relative[axis] / total_mass
                for axis in range(3)
            ],
        )
        numerator = sum(
            masses[p]
            * sum(
                (frame[p][axis_name] - exact_positions[p][axis]) ** 2
                for axis, axis_name in enumerate(("x", "y", "z"))
            )
            for p in (0, 1)
        )
        denominator = sum(
            masses[p] * sum(value * value for value in exact_positions[p])
            for p in (0, 1)
        )
        if denominator <= sys.float_info.epsilon:
            raise RuntimeError("theoretical position norm is too close to zero")
        position_errors.append(math.sqrt(numerator / denominator))
        numerical_separation = math.sqrt(
            sum(
                (frame[1][axis] - frame[0][axis]) ** 2
                for axis in ("x", "y", "z")
            )
        )
        separation_errors.append(abs(numerical_separation - SEPARATION) / SEPARATION)
    return position_errors, separation_errors


def run_case(profile, integrator, periods, steps_per_period, omega, runs_dir):
    steps = periods * steps_per_period
    dt = 2.0 * math.pi / omega / steps_per_period
    save_every = steps_per_period // 250
    stem = f"{integrator}_spp_{steps_per_period}"
    trajectory = runs_dir / f"{stem}_trajectory.csv"
    energy = runs_dir / f"{stem}_energy.csv"
    command = [
        str(BIN), "--N", "2", "--steps", str(steps), "--dt", f"{dt:.17g}",
        "--epsilon", f"{EPSILON:.17g}", "--G", f"{G:.17g}",
        "--init", "two-body-barycentric", "--integrator", integrator,
        "--save-every", str(save_every), "--output", str(trajectory),
        "--energy-output", str(energy),
    ]
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout, end="", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"{integrator} steps_per_period={steps_per_period} exited "
            f"with {completed.returncode}"
        )

    frames = read_trajectory(trajectory, 2, steps, save_every, dt)
    energy_rows = read_energy(energy, steps, save_every, dt)
    validate_matching_outputs(frames, energy_rows, trajectory)
    position_errors, separation_errors = position_and_separation_errors(frames, omega)
    position_window_steps = expected_saved_steps(steps_per_period, save_every)
    position_window_frames = len(position_window_steps)
    actual_window_steps = [
        int(frame[0]["step"]) for frame in frames[:position_window_frames]
    ]
    if actual_window_steps != position_window_steps:
        raise RuntimeError(
            f"unexpected first-period frame sequence for {integrator}: "
            f"{actual_window_steps}"
        )
    position_window_errors = position_errors[:position_window_frames]
    energy_errors = [row["relative_error"] for row in energy_rows]
    period = 2.0 * math.pi / omega
    return {
        "profile": profile, "initializer": "two-body-barycentric",
        "integrator": integrator, "G": G, "m1": M1, "m2": M2,
        "initial_separation": SEPARATION, "periods": periods,
        "steps_per_period": steps_per_period, "steps": steps, "dt": dt,
        "theoretical_period": period, "simulated_time": steps * dt,
        "epsilon": EPSILON, "save_every": save_every,
        "trajectory_frames": len(frames), "energy_samples": len(energy_rows),
        "position_error_window_periods": 1,
        "position_error_window_frames": position_window_frames,
        "rms_includes_initial": 0, "initial_energy": energy_rows[0]["total"],
        "max_relative_position_error": max(position_window_errors),
        "rms_relative_position_error": rms_without_initial(position_window_errors),
        "final_relative_position_error": position_window_errors[-1],
        "full_duration_max_relative_position_error": max(position_errors),
        "full_duration_rms_relative_position_error": rms_without_initial(position_errors),
        "full_duration_final_relative_position_error": position_errors[-1],
        "max_relative_energy_error": max(energy_errors),
        "rms_relative_energy_error": rms_without_initial(energy_errors),
        "final_relative_energy_error": energy_errors[-1],
        "max_relative_separation_error": max(separation_errors),
        "rms_relative_separation_error": rms_without_initial(separation_errors),
        "final_relative_separation_error": separation_errors[-1],
        "order_error_metric": "max_relative_position_error",
        "order_roundoff_floor": ROUND_OFF_FLOOR,
        "observed_position_order_max_from_previous": "",
    }


def add_observed_orders(rows):
    for integrator in INTEGRATORS:
        selected = sorted(
            (row for row in rows if row["integrator"] == integrator),
            key=lambda row: row["dt"], reverse=True,
        )
        for coarse, fine in zip(selected, selected[1:]):
            coarse_error = coarse["max_relative_position_error"]
            fine_error = fine["max_relative_position_error"]
            if min(coarse_error, fine_error) <= ROUND_OFF_FLOOR:
                continue
            fine["observed_position_order_max_from_previous"] = math.log(
                coarse_error / fine_error
            ) / math.log(coarse["dt"] / fine["dt"])


def write_summary(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, rows, profile, periods, period):
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True
    )
    expected_orders = {"euler": 1, "verlet": 2}
    for ax, integrator in zip(axes, INTEGRATORS):
        selected = sorted(
            (row for row in rows if row["integrator"] == integrator),
            key=lambda row: row["dt"],
        )
        dt_values = [row["dt"] for row in selected]
        errors = [row["max_relative_position_error"] for row in selected]
        ax.loglog(
            dt_values,
            errors,
            marker="o",
            label="Numerical error",
        )
        order = expected_orders[integrator]
        reference_dt = dt_values[-1]
        reference_error = errors[-1]
        guide = [reference_error * (dt / reference_dt) ** order for dt in dt_values]
        guide_label = (
            r"Reference $\propto \Delta t$"
            if order == 1 else r"Reference $\propto \Delta t^2$"
        )
        ax.loglog(dt_values, guide, linestyle="--", label=guide_label)
        ax.set_title("Euler" if integrator == "euler" else "Velocity Verlet")
        ax.set_xlabel(r"Time step $\Delta t$ [arbitrary time units]")
        ax.set_xticks(dt_values)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1e}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        finest = min(selected, key=lambda row: row["dt"])
        observed = float(finest["observed_position_order_max_from_previous"])
        ax.text(
            0.04, 0.95, rf"Finest-step order: $p \approx {observed:.2f}$",
            transform=ax.transAxes, va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    axes[0].set_ylabel("Max. mass-weighted relative position error")
    fig.suptitle(
        "Two-body timestep convergence ($\\epsilon = 0.01$)\n"
        "Maximum position error over the first orbital period "
        "$\\cdot$ log-log scale"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.chdir(ROOT)
    require_binary()
    config = PROFILES[args.profile]
    omega = theoretical_omega()
    period = 2.0 * math.pi / omega
    output_dir = RESULTS_ROOT / args.profile
    runs_dir = output_dir / "runs"
    summary = output_dir / f"barycentric_dt_study_{args.profile}.csv"
    plot = FIGURES_DIR / f"barycentric_dt_study_{args.profile}_v4.png"
    runs_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for integrator in INTEGRATORS:
        for steps_per_period in config["steps_per_period"]:
            if steps_per_period % 250 != 0:
                raise RuntimeError("steps_per_period must be divisible by 250")
            rows.append(
                run_case(
                    args.profile, integrator, config["periods"],
                    steps_per_period, omega, runs_dir,
                )
            )
    add_observed_orders(rows)
    write_summary(summary, rows)
    plot_summary(plot, rows, args.profile, config["periods"], period)

    for row in rows:
        print(
            f"{row['integrator']},steps_per_period={row['steps_per_period']},"
            f"max_position_error={row['max_relative_position_error']:.17g},"
            f"rms_position_error={row['rms_relative_position_error']:.17g},"
            f"final_position_error={row['final_relative_position_error']:.17g},"
            f"full_duration_max_position_error="
            f"{row['full_duration_max_relative_position_error']:.17g},"
            f"position_order={row['observed_position_order_max_from_previous']}"
        )
    print("rms_definition,excludes initial state (rms_includes_initial=0)")
    print("energy_order_interpretation,energy error is conservation diagnostic only")
    print(summary)
    print(plot)


if __name__ == "__main__":
    main()
