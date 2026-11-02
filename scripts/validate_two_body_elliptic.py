#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "nbody_matplotlib"))
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SERIAL_BIN = ROOT / "bin" / "nbody_serial"
RESULTS_ROOT = ROOT / "results" / "validation" / "two_body_elliptic_v1"
FIGURES_DIR = ROOT / "figures"
G, M0, M1, SEMI_MAJOR_AXIS, ECCENTRICITY = 1.0, 1.0, 1.0, 1.0, 0.5
TOTAL_MASS = M0 + M1
PERIAPSIS = SEMI_MAJOR_AXIS * (1.0 - ECCENTRICITY)
APOAPSIS = SEMI_MAJOR_AXIS * (1.0 + ECCENTRICITY)
PERIAPSIS_SPEED = math.sqrt(
    G * TOTAL_MASS * (1.0 + ECCENTRICITY)
    / (SEMI_MAJOR_AXIS * (1.0 - ECCENTRICITY))
)
PERIOD = 2.0 * math.pi * math.sqrt(SEMI_MAJOR_AXIS**3 / (G * TOTAL_MASS))
TRAJECTORY_FIELDS = ("step", "time", "particle", "x", "y", "z", "vx", "vy", "vz", "mass")
ENERGY_FIELDS = ("step", "time", "kinetic", "potential", "total", "relative_error")
TIME_FACTOR = 64.0
INITIAL_TOLERANCE = 128.0 * sys.float_info.epsilon
PROFILES = {
    "smoke": {"steps_per_period": 500, "periods": 2, "save_every": 5},
    "final": {"steps_per_period": 4000, "periods": 3, "save_every": 20},
}
FINAL_THRESHOLDS = {
    "max_abs_relative_energy_error": 2.0e-5,
    "period_relative_error": 5.0e-5,
    "a_relative_error": 5.0e-5,
    "e_relative_error": 5.0e-5,
    "max_relative_conic_residual": 1.0e-4,
    "max_relative_angular_momentum_drift": 1.0e-12,
    "max_center_of_mass_norm": 1.0e-12,
    "max_center_of_mass_speed": 1.0e-12,
    "max_total_momentum_norm": 1.0e-12,
    "max_abs_z": 1.0e-12,
    "max_abs_vz": 1.0e-12,
    "areal_rate_max_relative_deviation": 2.0e-3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an unsoftened barycentric elliptic two-body orbit.")
    parser.add_argument("--profile", required=True, choices=tuple(PROFILES))
    return parser.parse_args()


def expected_steps(steps, save_every):
    values = list(range(0, steps + 1, save_every))
    if values[-1] != steps:
        values.append(steps)
    return values


def parse_int(value, label, path):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise RuntimeError(f"non-integer {label} in {path}: {value}")
    return int(number)


def check_time(actual, step, dt, path):
    expected = step * dt
    tolerance = TIME_FACTOR * sys.float_info.epsilon * max(1.0, abs(expected))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"time mismatch at step {step} in {path}: {actual} != {expected}")


def read_trajectory(path, steps, save_every, dt):
    grouped = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRAJECTORY_FIELDS:
            raise RuntimeError(f"unexpected trajectory schema in {path}: {reader.fieldnames}")
        for raw in reader:
            row = {key: float(value) for key, value in raw.items()}
            if not all(math.isfinite(value) for value in row.values()):
                raise RuntimeError(f"non-finite trajectory row in {path}")
            step = parse_int(raw["step"], "step", path)
            particle = parse_int(raw["particle"], "particle ID", path)
            check_time(row["time"], step, dt, path)
            frame = grouped.setdefault(step, {})
            if particle in frame:
                raise RuntimeError(f"duplicate particle {particle} at step {step} in {path}")
            frame[particle] = row
    wanted = expected_steps(steps, save_every)
    if sorted(grouped) != wanted:
        raise RuntimeError(f"unexpected trajectory steps in {path}: {sorted(grouped)}")
    masses = None
    for step in wanted:
        if set(grouped[step]) != {0, 1}:
            raise RuntimeError(f"incomplete particle IDs at step {step} in {path}")
        current = [grouped[step][i]["mass"] for i in (0, 1)]
        if masses is None:
            masses = current
        elif current != masses:
            raise RuntimeError(f"particle mass changed at step {step} in {path}")
    return [grouped[step] for step in wanted]


def read_energy(path, steps, save_every, dt):
    grouped = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ENERGY_FIELDS:
            raise RuntimeError(f"unexpected energy schema in {path}: {reader.fieldnames}")
        for raw in reader:
            row = {key: float(value) for key, value in raw.items()}
            if not all(math.isfinite(value) for value in row.values()):
                raise RuntimeError(f"non-finite energy row in {path}")
            step = parse_int(raw["step"], "step", path)
            check_time(row["time"], step, dt, path)
            if step in grouped:
                raise RuntimeError(f"duplicate energy step {step} in {path}")
            grouped[step] = row
    wanted = expected_steps(steps, save_every)
    if sorted(grouped) != wanted:
        raise RuntimeError(f"unexpected energy steps in {path}: {sorted(grouped)}")
    return [grouped[step] for step in wanted]


def validate_alignment(frames, energies):
    if len(frames) != len(energies):
        raise RuntimeError("trajectory and energy frame counts differ")
    for frame, energy in zip(frames, energies):
        if (frame[0]["step"] != energy["step"] or
                frame[0]["time"] != energy["time"]):
            raise RuntimeError("trajectory and energy samples are not aligned")


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def cross(left, right):
    return (left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0])


def rms_without_initial(values):
    return math.sqrt(sum(value * value for value in values[1:]) / (len(values) - 1))


def quadratic_vertex(times, values, index):
    y0, y1, y2 = values[index - 1:index + 2]
    curvature = y0 - 2.0 * y1 + y2
    if curvature == 0.0:
        return times[index], y1
    offset = 0.5 * (y0 - y2) / curvature
    if abs(offset) > 1.0:
        return times[index], y1
    time = times[index] + offset * (times[index + 1] - times[index])
    value = y1 - (y2 - y0) ** 2 / (8.0 * curvature)
    return time, value


def unwrap_phases(phases):
    values = [phases[0]]
    for phase in phases[1:]:
        delta = phase - values[-1]
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        while delta > math.pi:
            delta -= 2.0 * math.pi
        values.append(values[-1] + delta)
    return values


def extrema_and_pericenters(times, radii):
    minima = [(times[0], radii[0])]
    maxima = []
    for index in range(1, len(radii) - 1):
        if radii[index] <= radii[index - 1] and radii[index] < radii[index + 1]:
            minima.append(quadratic_vertex(times, radii, index))
        if radii[index] >= radii[index - 1] and radii[index] > radii[index + 1]:
            maxima.append(quadratic_vertex(times, radii, index))
    if len(minima) < 2 or not maxima:
        raise RuntimeError("insufficient pericenter/apocenter samples")
    periods = [right[0] - left[0] for left, right in zip(minima, minima[1:])]
    peri, apo = min(value for _, value in minima), max(value for _, value in maxima)
    if not apo > peri:
        raise RuntimeError("pericenter and apocenter are not distinct")
    return peri, apo, periods


def initial_state_check(frame):
    expected = {
        0: (-0.25, 0.0, 0.0, 0.0, -PERIAPSIS_SPEED / 2.0, 0.0, 1.0),
        1: (0.25, 0.0, 0.0, 0.0, PERIAPSIS_SPEED / 2.0, 0.0, 1.0),
    }
    for particle in (0, 1):
        actual = tuple(frame[particle][key] for key in ("x", "y", "z", "vx", "vy", "vz", "mass"))
        for value, target in zip(actual, expected[particle]):
            tolerance = INITIAL_TOLERANCE * max(1.0, abs(target))
            if not math.isclose(value, target, rel_tol=0.0, abs_tol=tolerance):
                raise RuntimeError(f"incorrect initial state for particle {particle}: {actual}")


def compute_metrics(frames, energies):
    initial_state_check(frames[0])
    times = [row[0]["time"] for row in frames]
    rho, radii, angular_momenta = [], [], []
    com_norms, com_speeds, momenta, z_values, vz_values = [], [], [], [], []
    for frame in frames:
        masses = [frame[i]["mass"] for i in (0, 1)]
        positions = [[frame[i][axis] for axis in ("x", "y", "z")] for i in (0, 1)]
        velocities = [[frame[i][axis] for axis in ("vx", "vy", "vz")] for i in (0, 1)]
        relative = [positions[1][j] - positions[0][j] for j in range(3)]
        rho.append(relative)
        radii.append(norm(relative))
        com = [sum(masses[i] * positions[i][j] for i in (0, 1)) / sum(masses) for j in range(3)]
        com_v = [sum(masses[i] * velocities[i][j] for i in (0, 1)) / sum(masses) for j in range(3)]
        momentum = [sum(masses[i] * velocities[i][j] for i in (0, 1)) for j in range(3)]
        total_l = [sum(masses[i] * cross(positions[i], velocities[i])[j] for i in (0, 1)) for j in range(3)]
        com_norms.append(norm(com)); com_speeds.append(norm(com_v)); momenta.append(norm(momentum))
        angular_momenta.append(norm(total_l))
        z_values.append(max(abs(positions[i][2]) for i in (0, 1)))
        vz_values.append(max(abs(velocities[i][2]) for i in (0, 1)))
    signed_energy = [(row["total"] - energies[0]["total"]) / abs(energies[0]["total"]) for row in energies]
    theoretical_energy = -G * M0 * M1 / (2.0 * SEMI_MAJOR_AXIS)
    if not math.isclose(energies[0]["total"], theoretical_energy, rel_tol=0.0,
                        abs_tol=INITIAL_TOLERANCE):
        raise RuntimeError("numerical initial energy does not match the theoretical value")
    if not all(row["total"] < 0.0 for row in energies):
        raise RuntimeError("orbit is not bound: non-negative total energy")
    peri, apo, period_samples = extrema_and_pericenters(times, radii)
    period_numerical = statistics.mean(period_samples)
    a_numerical, e_numerical = (apo + peri) / 2.0, (apo - peri) / (apo + peri)
    phases = unwrap_phases([math.atan2(value[1], value[0]) for value in rho])
    p = SEMI_MAJOR_AXIS * (1.0 - ECCENTRICITY**2)
    conic = [abs(radius - p / (1.0 + ECCENTRICITY * math.cos(theta))) / SEMI_MAJOR_AXIS
             for radius, theta in zip(radii, phases)]
    areal = [norm(cross(rho[i], rho[i + 1])) / (2.0 * (times[i + 1] - times[i]))
             for i in range(len(rho) - 1)]
    areal_mean = statistics.mean(areal)
    metrics = {
        "theoretical_period": PERIOD, "theoretical_initial_energy": theoretical_energy,
        "numerical_initial_energy": energies[0]["total"],
        "max_abs_relative_energy_error": max(map(abs, signed_energy)),
        "rms_relative_energy_error": rms_without_initial(signed_energy),
        "final_signed_relative_energy_error": signed_energy[-1],
        "max_center_of_mass_norm": max(com_norms), "max_center_of_mass_speed": max(com_speeds),
        "max_total_momentum_norm": max(momenta), "initial_angular_momentum": angular_momenta[0],
        "max_relative_angular_momentum_drift": max(abs(value - angular_momenta[0]) / angular_momenta[0] for value in angular_momenta),
        "max_abs_z": max(z_values), "max_abs_vz": max(vz_values),
        "max_plane_deviation": max(z_values), "r_peri_numerical": peri, "r_apo_numerical": apo,
        "a_numerical": a_numerical, "e_numerical": e_numerical,
        "r_peri_relative_error": abs(peri - PERIAPSIS) / PERIAPSIS,
        "r_apo_relative_error": abs(apo - APOAPSIS) / APOAPSIS,
        "a_relative_error": abs(a_numerical - SEMI_MAJOR_AXIS) / SEMI_MAJOR_AXIS,
        "e_relative_error": abs(e_numerical - ECCENTRICITY) / ECCENTRICITY,
        "max_relative_conic_residual": max(conic), "areal_rate_mean": areal_mean,
        "areal_rate_sample_std": statistics.stdev(areal),
        "areal_rate_coefficient_of_variation": statistics.stdev(areal) / areal_mean,
        "areal_rate_max_relative_deviation": max(abs(value - areal_mean) / areal_mean for value in areal),
        "period_numerical": period_numerical,
        "period_relative_error": abs(period_numerical - PERIOD) / PERIOD,
        "pericenter_intervals": len(period_samples), "rms_includes_initial": 0,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite metric: {metrics}")
    if metrics["max_abs_z"] > INITIAL_TOLERANCE or metrics["max_abs_vz"] > INITIAL_TOLERANCE:
        raise RuntimeError("trajectory left the expected orbital plane")
    return metrics, {"times": times, "rho": rho, "energy": signed_energy, "areal": areal}


def validate_final_thresholds(metrics):
    failures = [
        f"{name}={metrics[name]:.17g} must be < {limit:.17g}"
        for name, limit in FINAL_THRESHOLDS.items()
        if metrics[name] >= limit
    ]
    if failures:
        raise RuntimeError("final numerical thresholds failed: " + "; ".join(failures))


def write_outputs(summary, metrics, series, frames, profile, steps, dt):
    row = {"profile": profile, "initializer": "two-body-elliptic", "G": G, "epsilon": 0.0,
           "periods": PROFILES[profile]["periods"],
           "steps_per_period": PROFILES[profile]["steps_per_period"], "steps": steps,
           "dt": dt, "save_every": PROFILES[profile]["save_every"],
           "frames": len(frames), **metrics}
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for particle, label in ((0, "body 0"), (1, "body 1")):
        axes[0].plot([frame[particle]["x"] for frame in frames], [frame[particle]["y"] for frame in frames], label=label)
        axes[0].scatter(frames[0][particle]["x"], frames[0][particle]["y"], s=20)
    axes[0].scatter(0.0, 0.0, marker="+", color="black", label="COM / focus")
    axes[0].set_aspect("equal", adjustable="box"); axes[0].set_title("First law: barycentric ellipses")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].legend()
    periods = [time / PERIOD for time in series["times"]]
    axes[1].plot(periods, series["energy"]); axes[1].set_title("Velocity Verlet energy conservation")
    axes[1].set_xlabel("time [orbital periods]"); axes[1].set_ylabel("(E(t)-E(0))/|E(0)|")
    normalized_areal = [value / metrics["areal_rate_mean"] for value in series["areal"]]
    axes[2].plot(periods[:-1], normalized_areal); axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].set_title("Second law: discrete areal rate")
    axes[2].set_xlabel("time [orbital periods]"); axes[2].set_ylabel("areal rate / mean")
    for axis in axes: axis.grid(True, alpha=0.3)
    fig.suptitle(f"Unsoftened elliptic two-body validation ({profile})"); fig.tight_layout()
    figure = FIGURES_DIR / f"two_body_elliptic_{profile}.png"; fig.savefig(figure, dpi=150); plt.close(fig)
    return row, figure


def main():
    args = parse_args(); config = PROFILES[args.profile]
    if not SERIAL_BIN.exists():
        raise RuntimeError(f"missing executable: {SERIAL_BIN}; run `make` first")
    output_dir = RESULTS_ROOT / args.profile; output_dir.mkdir(parents=True, exist_ok=True); FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    steps = config["periods"] * config["steps_per_period"]; dt = PERIOD / config["steps_per_period"]
    trajectory = output_dir / "trajectory.csv"; energy = output_dir / "energy.csv"
    command = [str(SERIAL_BIN), "--N", "2", "--steps", str(steps), "--dt", f"{dt:.17g}",
               "--epsilon", "0", "--G", "1", "--init", "two-body-elliptic", "--integrator", "verlet",
               "--save-every", str(config["save_every"]), "--output", str(trajectory), "--energy-output", str(energy)]
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"simulation failed ({completed.returncode}):\n{completed.stdout}{completed.stderr}")
    frames = read_trajectory(trajectory, steps, config["save_every"], dt)
    energies = read_energy(energy, steps, config["save_every"], dt)
    validate_alignment(frames, energies)
    metrics, series = compute_metrics(frames, energies)
    if args.profile == "final":
        validate_final_thresholds(metrics)
    row, figure = write_outputs(output_dir / "summary.csv", metrics, series, frames, args.profile, steps, dt)
    for key, value in row.items(): print(f"{key},{value:.17g}" if isinstance(value, float) else f"{key},{value}")
    print("third_law_scope,period relation verified for this simulated semi-major axis only")
    print(output_dir / "summary.csv"); print(figure)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr); raise SystemExit(1)
