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


ROOT = Path(__file__).resolve().parents[1]
SERIAL_BIN = ROOT / "bin" / "nbody_serial"
RESULTS_ROOT = ROOT / "results" / "validation" / "epsilon_study_v3"
FIGURES_DIR = ROOT / "figures"

G = 1.0
M1 = 1.0
M2 = 1.0e-3
SEPARATION = 1.0
REFERENCE_EPSILON = 0.01
TIME_TOLERANCE_FACTOR = 64.0
TRAJECTORY_FIELDS = (
    "step", "time", "particle", "x", "y", "z", "vx", "vy", "vz", "mass"
)
ENERGY_FIELDS = ("step", "time", "kinetic", "potential", "total", "relative_error")
PROFILES = {
    "smoke": {
        "epsilon_values": (0.0, 0.01, 0.1),
        "steps": 500,
        "samples": 100,
    },
    "final": {
        "epsilon_values": (0.0, 0.001, 0.01, 0.05, 0.1, 0.2),
        "steps": 4000,
        "samples": 400,
    },
}

SUMMARY_FIELDS = [
    "profile",
    "epsilon",
    "integrator",
    "initializer",
    "G",
    "m1",
    "m2",
    "initial_separation",
    "reference_epsilon_for_common_duration",
    "steps",
    "dt",
    "simulated_time",
    "save_every",
    "energy_samples",
    "rms_includes_initial",
    "theoretical_omega",
    "theoretical_period",
    "simulated_periods",
    "numerical_period",
    "numerical_period_relative_error",
    "initial_light_particle_circular_speed",
    "softened_force_ratio_at_initial_separation",
    "relative_force_reduction_at_initial_separation",
    "initial_energy",
    "max_relative_energy_error",
    "rms_relative_energy_error",
    "final_relative_energy_error",
    "max_relative_separation_error",
    "final_relative_separation_error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Separate numerical and physical effects of gravitational softening."
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    return parser.parse_args()


def require_binary():
    if not SERIAL_BIN.exists():
        raise RuntimeError(f"missing executable: {SERIAL_BIN}; run `make` first")


def epsilon_tag(epsilon):
    return f"{epsilon:g}".replace(".", "p")


def theoretical_orbit(epsilon):
    softened = (SEPARATION * SEPARATION + epsilon * epsilon) ** 1.5
    omega = math.sqrt(G * (M1 + M2) / softened)
    period = 2.0 * math.pi / omega
    light_radius = M1 * SEPARATION / (M1 + M2)
    return omega, period, omega * light_radius


def softened_force_ratio(epsilon):
    numerator = SEPARATION ** 3
    denominator = (SEPARATION * SEPARATION + epsilon * epsilon) ** 1.5
    return numerator / denominator


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


def read_frames(path, particle_count, steps, save_every, dt):
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


def vector_norm(values):
    return math.sqrt(sum(value * value for value in values))


def unwrap_phases(phases):
    unwrapped = [phases[0]]
    previous = phases[0]
    for phase in phases[1:]:
        delta = phase - previous
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        while delta > math.pi:
            delta -= 2.0 * math.pi
        unwrapped.append(unwrapped[-1] + delta)
        previous = phase
    return unwrapped


def estimate_period(times, phases):
    mean_time = sum(times) / len(times)
    mean_phase = sum(phases) / len(phases)
    covariance = sum(
        (time - mean_time) * (phase - mean_phase)
        for time, phase in zip(times, phases)
    )
    variance = sum((time - mean_time) ** 2 for time in times)
    if variance == 0.0:
        raise RuntimeError("cannot estimate period from zero time variance")
    angular_frequency = covariance / variance
    if not math.isfinite(angular_frequency) or angular_frequency <= 0.0:
        raise RuntimeError(f"invalid fitted angular frequency: {angular_frequency}")
    return 2.0 * math.pi / angular_frequency


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def rms_without_initial(values):
    if len(values) < 2:
        raise RuntimeError("RMS requires at least one sample after the initial state")
    return rms(values[1:])


def simulator_command(epsilon, steps, dt, save_every, trajectory, energy):
    return [
        str(SERIAL_BIN),
        "--N", "2",
        "--steps", str(steps),
        "--dt", f"{dt:.17g}",
        "--epsilon", f"{epsilon:.17g}",
        "--G", f"{G:.17g}",
        "--init", "two-body-barycentric",
        "--integrator", "verlet",
        "--save-every", str(save_every),
        "--output", str(trajectory),
        "--energy-output", str(energy),
    ]


def compute_metrics(profile, epsilon, steps, dt, save_every, frames, energy_rows):
    omega, period, light_speed = theoretical_orbit(epsilon)
    times = [frame[0]["time"] for frame in frames]
    phases = []
    separation_errors = []
    for frame in frames:
        relative = [
            frame[1][axis] - frame[0][axis] for axis in ("x", "y", "z")
        ]
        phases.append(math.atan2(relative[1], relative[0]))
        separation_errors.append(
            abs(vector_norm(relative) - SEPARATION) / SEPARATION
        )

    energy_errors = [row["relative_error"] for row in energy_rows]
    ratio = softened_force_ratio(epsilon)
    row = {
        "profile": profile,
        "epsilon": epsilon,
        "integrator": "verlet",
        "initializer": "two-body-barycentric",
        "G": G,
        "m1": M1,
        "m2": M2,
        "initial_separation": SEPARATION,
        "reference_epsilon_for_common_duration": REFERENCE_EPSILON,
        "steps": steps,
        "dt": dt,
        "simulated_time": steps * dt,
        "save_every": save_every,
        "energy_samples": len(energy_rows),
        "rms_includes_initial": 0,
        "theoretical_omega": omega,
        "theoretical_period": period,
        "simulated_periods": steps * dt / period,
        "numerical_period": estimate_period(times, unwrap_phases(phases)),
        "initial_light_particle_circular_speed": light_speed,
        "softened_force_ratio_at_initial_separation": ratio,
        "relative_force_reduction_at_initial_separation": 1.0 - ratio,
        "initial_energy": energy_rows[0]["total"],
        "max_relative_energy_error": max(energy_errors),
        "rms_relative_energy_error": rms_without_initial(energy_errors),
        "final_relative_energy_error": energy_errors[-1],
        "max_relative_separation_error": max(separation_errors),
        "final_relative_separation_error": separation_errors[-1],
    }
    row["numerical_period_relative_error"] = abs(
        row["numerical_period"] - period
    ) / period
    numeric_values = [value for value in row.values() if not isinstance(value, str)]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError(f"non-finite epsilon-study metric: {row}")
    return row


def run_case(profile, epsilon, config, common_dt, runs_dir):
    steps = config["steps"]
    save_every = max(1, steps // config["samples"])
    tag = epsilon_tag(epsilon)
    trajectory = runs_dir / f"epsilon_{tag}_trajectory.csv"
    energy = runs_dir / f"epsilon_{tag}_energy.csv"
    command = simulator_command(
        epsilon, steps, common_dt, save_every, trajectory, energy
    )
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout, end="", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"epsilon={epsilon:g} exited with {completed.returncode}")

    frames = read_frames(trajectory, 2, steps, save_every, common_dt)
    energy_rows = read_energy(energy, steps, save_every, common_dt)
    validate_matching_outputs(frames, energy_rows, trajectory)
    return compute_metrics(
        profile, epsilon, steps, common_dt, save_every, frames, energy_rows
    )


def write_summary(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, rows, profile):
    epsilon = [row["epsilon"] for row in rows]
    dt = rows[0]["dt"]
    dt_exponent = math.floor(math.log10(dt))
    dt_mantissa = dt / 10.0 ** dt_exponent
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )

    axes[0].plot(
        epsilon,
        [100.0 * row["relative_force_reduction_at_initial_separation"] for row in rows],
        marker="o",
        label="Force reduction",
    )
    axes[0].plot(
        epsilon,
        [100.0 * (row["theoretical_period"] / rows[0]["theoretical_period"] - 1.0)
         for row in rows],
        marker="s",
        label="Orbital-period increase",
    )
    axes[0].axvline(0.01, color="gray", linestyle="--", linewidth=1)
    axes[0].text(
        0.01, 0.96, r"Selected $\epsilon = 0.01$",
        transform=axes[0].get_xaxis_transform(), rotation=90,
        va="top", ha="right", color="dimgray",
    )
    axes[0].set_title("Physical effect at the initial separation")
    axes[0].set_xlabel(r"Softening length $\epsilon$")
    axes[0].set_ylabel("Relative change [%]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for key, label in (
        ("max_relative_energy_error", "Maximum"),
        ("rms_relative_energy_error", "RMS"),
        ("final_relative_energy_error", "Final"),
    ):
        positive = [(row["epsilon"], row[key]) for row in rows if row[key] > 0.0]
        axes[1].plot(
            [item[0] for item in positive], [item[1] for item in positive],
            marker="o", label=label,
        )
    axes[1].set_yscale("log")
    axes[1].set_title("Energy conservation with Velocity Verlet")
    axes[1].set_xlabel(r"Softening length $\epsilon$")
    axes[1].set_ylabel("Relative energy error")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        "Effect of gravitational softening\n"
        rf"Circular two-body system, Velocity Verlet, $\Delta t = "
        rf"{dt_mantissa:.3g}\times10^{{{dt_exponent}}}$"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.chdir(ROOT)
    require_binary()
    config = PROFILES[args.profile]
    reference_period = theoretical_orbit(REFERENCE_EPSILON)[1]
    common_dt = reference_period / config["steps"]
    output_dir = RESULTS_ROOT / args.profile
    runs_dir = output_dir / "runs"
    summary = output_dir / f"epsilon_study_{args.profile}.csv"
    plot = FIGURES_DIR / f"epsilon_study_{args.profile}_v3.png"
    runs_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        run_case(args.profile, epsilon, config, common_dt, runs_dir)
        for epsilon in config["epsilon_values"]
    ]
    write_summary(summary, rows)
    plot_summary(plot, rows, args.profile)

    for row in rows:
        print(
            f"epsilon={row['epsilon']:.17g},force_reduction="
            f"{row['relative_force_reduction_at_initial_separation']:.17g},"
            f"max_energy_error={row['max_relative_energy_error']:.17g},"
            f"rms_energy_error={row['rms_relative_energy_error']:.17g},"
            f"final_energy_error={row['final_relative_energy_error']:.17g}"
        )
    print("interpretation,numerical error and physical force change are separate metrics")
    print("selection,no universally optimal epsilon is inferred")
    print(summary)
    print(plot)


if __name__ == "__main__":
    main()
