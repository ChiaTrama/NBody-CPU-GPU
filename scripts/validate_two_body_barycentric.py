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
RESULTS_DIR = ROOT / "results" / "validation"
FIGURES_DIR = ROOT / "figures"
TRAJECTORY_CSV = RESULTS_DIR / "two_body_barycentric_trajectory.csv"
ENERGY_CSV = RESULTS_DIR / "two_body_barycentric_energy.csv"
SUMMARY_CSV = RESULTS_DIR / "two_body_barycentric_validation.csv"
ORBIT_PNG = FIGURES_DIR / "two_body_barycentric_orbit.png"
LONG_RESULTS_DIR = RESULTS_DIR / "two_body_barycentric_long_term_v2"

G = 1.0
EPSILON = 0.01
M1 = 1.0
M2 = 1.0e-3
SEPARATION = 1.0
STEPS_PER_PERIOD = 4000
TIME_TOLERANCE_FACTOR = 64.0
TRAJECTORY_FIELDS = (
    "step", "time", "particle", "x", "y", "z", "vx", "vy", "vz", "mass"
)
ENERGY_FIELDS = ("step", "time", "kinetic", "potential", "total", "relative_error")

THRESHOLDS = {
    "max_center_of_mass_drift": 1.0e-12,
    "max_total_momentum_drift": 1.0e-12,
    "final_separation_relative_error": 1.0e-6,
    "final_state_distance": 1.0e-4,
    "final_relative_energy_error": 1.0e-10,
    "numerical_period_relative_error": 1.0e-5,
}

PROFILE_CONFIG = {
    "smoke": {"periods": 1, "steps_per_period": 250, "samples_per_period": 50},
    "long-term": {
        "periods": 10,
        "steps_per_period": 2000,
        "samples_per_period": 100,
    },
}

LONG_SUMMARY_FIELDS = [
    "profile",
    "integrator",
    "initializer",
    "G",
    "epsilon",
    "m1",
    "m2",
    "initial_separation",
    "periods",
    "steps_per_period",
    "steps",
    "dt",
    "simulated_time",
    "save_every",
    "samples",
    "rms_includes_initial",
    "initial_kinetic",
    "initial_potential",
    "initial_total_energy",
    "max_relative_energy_error",
    "rms_relative_energy_error",
    "final_relative_energy_error",
    "max_relative_separation_error",
    "rms_relative_separation_error",
    "final_relative_separation_error",
    "max_center_of_mass_drift",
    "max_center_of_mass_speed",
    "max_total_momentum_norm",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the softened barycentric two-body orbit."
    )
    parser.add_argument(
        "--profile",
        choices=("one-period", "smoke", "long-term"),
        default="one-period",
        help="one-period preserves the historical default validation",
    )
    return parser.parse_args()


def require_binary():
    if not SERIAL_BIN.exists():
        raise RuntimeError(f"missing executable: {SERIAL_BIN}; run `make` first")


def theoretical_omega():
    softened = (SEPARATION * SEPARATION + EPSILON * EPSILON) ** 1.5
    return math.sqrt(G * (M1 + M2) / softened)


def simulator_command(period):
    dt = period / STEPS_PER_PERIOD
    return [
        str(SERIAL_BIN),
        "--N", "2",
        "--steps", str(STEPS_PER_PERIOD),
        "--dt", f"{dt:.17g}",
        "--epsilon", str(EPSILON),
        "--G", str(G),
        "--init", "two-body-barycentric",
        "--integrator", "verlet",
        "--save-every", "1",
        "--output", str(TRAJECTORY_CSV),
        "--energy-output", str(ENERGY_CSV),
    ]


def parse_relative_energy_error(stdout):
    for line in stdout.splitlines():
        key, separator, value = line.partition(",")
        if separator and key == "relative_energy_error":
            result = float(value)
            if math.isfinite(result):
                return result
    raise RuntimeError("serial output lacks a finite relative_energy_error")


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
            parsed = {key: float(value) for key, value in raw.items()}
            if not all(math.isfinite(value) for value in parsed.values()):
                raise RuntimeError(f"non-finite trajectory row in {path}: {raw}")
            step = parse_integer(raw["step"], "step", path)
            particle = parse_integer(raw["particle"], "particle ID", path)
            validate_time(parsed["time"], step, dt, path)
            frame = frames.setdefault(step, {})
            if particle in frame:
                raise RuntimeError(f"duplicate particle {particle} at step {step} in {path}")
            frame[particle] = parsed

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
            parsed = {key: float(value) for key, value in raw.items()}
            if not all(math.isfinite(value) for value in parsed.values()):
                raise RuntimeError(f"non-finite energy row in {path}: {raw}")
            step = parse_integer(raw["step"], "step", path)
            validate_time(parsed["time"], step, dt, path)
            if step in rows:
                raise RuntimeError(f"duplicate energy row at step {step} in {path}")
            rows[step] = parsed
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
    for phase in phases[1:]:
        delta = phase - phases[len(unwrapped) - 1]
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        while delta > math.pi:
            delta -= 2.0 * math.pi
        unwrapped.append(unwrapped[-1] + delta)
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
    angular_slope = covariance / variance
    if not math.isfinite(angular_slope) or angular_slope <= 0.0:
        raise RuntimeError(f"invalid fitted angular frequency: {angular_slope}")
    return 2.0 * math.pi / angular_slope


def compute_metrics(frames, theoretical_period, energy_error):
    total_mass = M1 + M2
    initial = frames[0]
    initial_com = [
        (M1 * initial[0][axis] + M2 * initial[1][axis]) / total_mass
        for axis in ("x", "y", "z")
    ]
    initial_momentum = [
        M1 * initial[0][axis] + M2 * initial[1][axis]
        for axis in ("vx", "vy", "vz")
    ]

    max_com_drift = 0.0
    max_momentum_drift = 0.0
    times = []
    phases = []
    for frame in frames:
        com = [
            (M1 * frame[0][axis] + M2 * frame[1][axis]) / total_mass
            for axis in ("x", "y", "z")
        ]
        momentum = [
            M1 * frame[0][axis] + M2 * frame[1][axis]
            for axis in ("vx", "vy", "vz")
        ]
        max_com_drift = max(
            max_com_drift,
            vector_norm([value - reference for value, reference in zip(com, initial_com)]),
        )
        max_momentum_drift = max(
            max_momentum_drift,
            vector_norm(
                [value - reference for value, reference in zip(momentum, initial_momentum)]
            ),
        )
        dx = frame[1]["x"] - frame[0]["x"]
        dy = frame[1]["y"] - frame[0]["y"]
        times.append(frame[0]["time"])
        phases.append(math.atan2(dy, dx))

    final = frames[-1]
    relative_position = [
        final[1][axis] - final[0][axis] for axis in ("x", "y", "z")
    ]
    final_separation = vector_norm(relative_position)
    final_state = [
        final[particle][axis]
        for particle in (0, 1)
        for axis in ("x", "y", "z", "vx", "vy", "vz")
    ]
    initial_state = [
        initial[particle][axis]
        for particle in (0, 1)
        for axis in ("x", "y", "z", "vx", "vy", "vz")
    ]
    numerical_period = estimate_period(times, unwrap_phases(phases))

    metrics = {
        "theoretical_period": theoretical_period,
        "simulated_time": times[-1],
        "numerical_period": numerical_period,
        "numerical_period_relative_error": abs(numerical_period - theoretical_period)
        / theoretical_period,
        "max_center_of_mass_drift": max_com_drift,
        "max_total_momentum_drift": max_momentum_drift,
        "final_separation_relative_error": abs(final_separation - SEPARATION)
        / SEPARATION,
        "final_state_distance": vector_norm(
            [value - reference for value, reference in zip(final_state, initial_state)]
        ),
        "final_relative_energy_error": energy_error,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite validation metric: {metrics}")
    return metrics


def write_summary(metrics, failures):
    row = dict(metrics)
    row["status"] = "PASS" if not failures else "FAIL"
    row["details"] = "; ".join(failures)
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def plot_orbits(frames):
    fig, ax = plt.subplots(figsize=(6, 6))
    for particle, label in ((0, "m1 = 1"), (1, "m2 = 1e-3")):
        ax.plot(
            [frame[particle]["x"] for frame in frames],
            [frame[particle]["y"] for frame in frames],
            label=label,
        )
        ax.scatter(frames[0][particle]["x"], frames[0][particle]["y"], s=25)
    ax.scatter(0.0, 0.0, marker="+", color="black", label="center of mass")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ORBIT_PNG, dpi=150)
    plt.close(fig)


def long_term_command(integrator, config, period, trajectory, energy):
    steps = config["periods"] * config["steps_per_period"]
    dt = period / config["steps_per_period"]
    save_every = config["steps_per_period"] // config["samples_per_period"]
    return [
        str(SERIAL_BIN),
        "--N", "2",
        "--steps", str(steps),
        "--dt", f"{dt:.17g}",
        "--epsilon", f"{EPSILON:.17g}",
        "--G", f"{G:.17g}",
        "--init", "two-body-barycentric",
        "--integrator", integrator,
        "--save-every", str(save_every),
        "--output", str(trajectory),
        "--energy-output", str(energy),
    ]


def long_term_series(frames, energy_rows):
    if len(frames) != len(energy_rows):
        raise RuntimeError(
            f"trajectory/energy sample mismatch: {len(frames)} != {len(energy_rows)}"
        )

    initial = frames[0]
    masses = [initial[particle]["mass"] for particle in (0, 1)]
    total_mass = sum(masses)
    initial_com = [
        sum(masses[p] * initial[p][axis] for p in (0, 1)) / total_mass
        for axis in ("x", "y", "z")
    ]
    series = {
        "time": [],
        "kinetic": [],
        "potential": [],
        "total": [],
        "energy_error": [],
        "separation_error": [],
        "com_drift": [],
        "com_speed": [],
        "momentum": [],
    }

    for frame, energy in zip(frames, energy_rows):
        if int(frame[0]["step"]) != int(energy["step"]):
            raise RuntimeError("trajectory and energy samples have different steps")
        com = [
            sum(masses[p] * frame[p][axis] for p in (0, 1)) / total_mass
            for axis in ("x", "y", "z")
        ]
        com_velocity = [
            sum(masses[p] * frame[p][axis] for p in (0, 1)) / total_mass
            for axis in ("vx", "vy", "vz")
        ]
        momentum = [value * total_mass for value in com_velocity]
        separation = vector_norm(
            [frame[1][axis] - frame[0][axis] for axis in ("x", "y", "z")]
        )

        series["time"].append(energy["time"])
        series["kinetic"].append(energy["kinetic"])
        series["potential"].append(energy["potential"])
        series["total"].append(energy["total"])
        series["energy_error"].append(energy["relative_error"])
        series["separation_error"].append(abs(separation - SEPARATION) / SEPARATION)
        series["com_drift"].append(
            vector_norm([value - base for value, base in zip(com, initial_com)])
        )
        series["com_speed"].append(vector_norm(com_velocity))
        series["momentum"].append(vector_norm(momentum))
    return series


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def rms_without_initial(values):
    if len(values) < 2:
        raise RuntimeError("RMS requires at least one sample after the initial state")
    return rms(values[1:])


def long_term_metrics(profile, integrator, config, period, series):
    steps = config["periods"] * config["steps_per_period"]
    row = {
        "profile": profile,
        "integrator": integrator,
        "initializer": "two-body-barycentric",
        "G": G,
        "epsilon": EPSILON,
        "m1": M1,
        "m2": M2,
        "initial_separation": SEPARATION,
        "periods": config["periods"],
        "steps_per_period": config["steps_per_period"],
        "steps": steps,
        "dt": period / config["steps_per_period"],
        "simulated_time": series["time"][-1],
        "save_every": config["steps_per_period"] // config["samples_per_period"],
        "samples": len(series["time"]),
        "rms_includes_initial": 0,
        "initial_kinetic": series["kinetic"][0],
        "initial_potential": series["potential"][0],
        "initial_total_energy": series["total"][0],
        "max_relative_energy_error": max(series["energy_error"]),
        "rms_relative_energy_error": rms_without_initial(series["energy_error"]),
        "final_relative_energy_error": series["energy_error"][-1],
        "max_relative_separation_error": max(series["separation_error"]),
        "rms_relative_separation_error": rms_without_initial(series["separation_error"]),
        "final_relative_separation_error": series["separation_error"][-1],
        "max_center_of_mass_drift": max(series["com_drift"]),
        "max_center_of_mass_speed": max(series["com_speed"]),
        "max_total_momentum_norm": max(series["momentum"]),
    }
    numeric_values = [value for value in row.values() if not isinstance(value, str)]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError(f"non-finite long-term metric: {row}")
    return row


def write_long_term_summary(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def scientific_math(value, decimals=2):
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0 ** exponent
    return rf"${mantissa:.{decimals}f} \times 10^{{{exponent}}}$"


def plot_long_term_energy(path, series_by_integrator, period, profile):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = {"euler": "tab:blue", "verlet": "tab:orange"}
    labels = {"euler": "Forward Euler", "verlet": "Velocity Verlet"}
    maxima = {}
    for integrator in ("euler", "verlet"):
        series = series_by_integrator[integrator]
        periods = [time / period for time in series["time"]]
        positive = [(time, error) for time, error in zip(periods, series["energy_error"])
                    if error > 0.0]
        ax.plot(
            [item[0] for item in positive], [item[1] for item in positive],
            color=colors[integrator], label=labels[integrator],
        )
        maxima[integrator] = max(series["energy_error"])
    ax.set_xlabel("Time [orbital periods]")
    ax.set_ylabel(r"Relative energy error  $|E(t) - E(0)| / |E(0)|$")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(
        "Two-body energy conservation over 10 orbital periods\n"
        rf"$\epsilon = 0.01$, $\Delta t = "
        rf"{scientific_math(period / 2000)[1:-1]}$ (2000 steps per period)",
    )
    ax.text(
        0.98, 0.68,
        f"Max Euler: {scientific_math(maxima['euler'])}\n"
        f"Max Verlet: {scientific_math(maxima['verlet'])}",
        transform=ax.transAxes, ha="right", va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_long_term_diagnostics(path, series_by_integrator, period, profile):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = (
        ("energy_error", "Relative energy error", r"$|E(t) - E(0)| / |E(0)|$"),
        ("separation_error", "Relative orbital-radius error", r"$|r(t) - r(0)| / r(0)$"),
        ("com_drift", "Center-of-mass drift", r"$|R_{CM}(t) - R_{CM}(0)|$"),
        ("momentum", "Total momentum norm", r"$|P(t)|$"),
    )
    colors = {"euler": "tab:blue", "verlet": "tab:orange"}
    labels = {"euler": "Forward Euler", "verlet": "Velocity Verlet"}
    for panel_index, (ax, (key, title, ylabel)) in enumerate(zip(axes.flat, panels)):
        for integrator in ("euler", "verlet"):
            series = series_by_integrator[integrator]
            periods = [time / period for time in series["time"]]
            positive = [(time, value) for time, value in zip(periods, series[key])
                        if value > 0.0]
            ax.plot(
                [item[0] for item in positive], [item[1] for item in positive],
                color=colors[integrator], label=labels[integrator],
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if panel_index < 2:
            maxima = {
                integrator: max(series_by_integrator[integrator][key])
                for integrator in ("euler", "verlet")
            }
            ax.text(
                0.98, 0.64,
                f"Max Euler: {scientific_math(maxima['euler'])}\n"
                f"Max Verlet: {scientific_math(maxima['verlet'])}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
    axes[1, 0].set_xlabel("Time [orbital periods]")
    axes[1, 1].set_xlabel("Time [orbital periods]")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.89))
    fig.suptitle("Two-body long-term conservation diagnostics", y=0.98)
    fig.text(
        0.5, 0.935,
        r"$\epsilon = 0.01$, 10 orbital periods, 2000 steps per period",
        ha="center",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_extended_profile(profile, period):
    config = PROFILE_CONFIG[profile]
    output_dir = LONG_RESULTS_DIR / profile
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    series_by_integrator = {}

    for integrator in ("euler", "verlet"):
        trajectory = output_dir / f"{integrator}_trajectory.csv"
        energy = output_dir / f"{integrator}_energy.csv"
        command = long_term_command(integrator, config, period, trajectory, energy)
        print("running:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout, end="", file=sys.stderr)
            print(completed.stderr, end="", file=sys.stderr)
            raise SystemExit(completed.returncode)

        steps = config["periods"] * config["steps_per_period"]
        dt = period / config["steps_per_period"]
        save_every = config["steps_per_period"] // config["samples_per_period"]
        frames = read_frames(trajectory, 2, steps, save_every, dt)
        energy_rows = read_energy(energy, steps, save_every, dt)
        validate_matching_outputs(frames, energy_rows, trajectory)
        series = long_term_series(frames, energy_rows)
        series_by_integrator[integrator] = series
        rows.append(long_term_metrics(profile, integrator, config, period, series))

    summary = output_dir / f"two_body_barycentric_{profile}_summary.csv"
    energy_plot = FIGURES_DIR / f"two_body_barycentric_{profile}_energy_v2.png"
    diagnostics_plot = FIGURES_DIR / f"two_body_barycentric_{profile}_diagnostics_v2.png"
    write_long_term_summary(summary, rows)
    plot_long_term_energy(energy_plot, series_by_integrator, period, profile)
    plot_long_term_diagnostics(diagnostics_plot, series_by_integrator, period, profile)

    for row in rows:
        print(
            f"{row['integrator']},max_energy_error="
            f"{row['max_relative_energy_error']:.17g},rms_energy_error="
            f"{row['rms_relative_energy_error']:.17g},final_energy_error="
            f"{row['final_relative_energy_error']:.17g}"
        )
    euler = next(row for row in rows if row["integrator"] == "euler")
    verlet = next(row for row in rows if row["integrator"] == "verlet")
    if euler["max_relative_energy_error"] > verlet["max_relative_energy_error"]:
        print("energy_comparison,euler_max_exceeds_verlet_max")
    else:
        print(f"warning,{profile} data do not show larger Euler maximum energy error")
    print(summary)
    print(energy_plot)
    print(diagnostics_plot)


def run_one_period():
    os.chdir(ROOT)
    require_binary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    omega = theoretical_omega()
    theoretical_period = 2.0 * math.pi / omega
    command = simulator_command(theoretical_period)
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True
    )
    if completed.returncode != 0:
        print(completed.stdout, end="", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(completed.returncode)

    energy_error = parse_relative_energy_error(completed.stdout)
    dt = theoretical_period / STEPS_PER_PERIOD
    frames = read_frames(TRAJECTORY_CSV, 2, STEPS_PER_PERIOD, 1, dt)
    energy_rows = read_energy(ENERGY_CSV, STEPS_PER_PERIOD, 1, dt)
    validate_matching_outputs(frames, energy_rows, TRAJECTORY_CSV)
    metrics = compute_metrics(frames, theoretical_period, energy_error)
    failures = [
        f"{name}={metrics[name]:.17g} exceeds {limit:.17g}"
        for name, limit in THRESHOLDS.items()
        if metrics[name] > limit
    ]
    write_summary(metrics, failures)
    plot_orbits(frames)

    for name, value in metrics.items():
        print(f"{name},{value:.17g}")
    for name, limit in THRESHOLDS.items():
        print(f"threshold_{name},{limit:.17g}")
    print(f"validation_status,{'PASS' if not failures else 'FAIL'}")
    print(SUMMARY_CSV)
    print(ORBIT_PNG)
    if failures:
        for failure in failures:
            print(f"validation failure: {failure}", file=sys.stderr)
        raise SystemExit(1)


def main():
    args = parse_args()
    os.chdir(ROOT)
    require_binary()
    if args.profile == "one-period":
        run_one_period()
        return
    period = 2.0 * math.pi / theoretical_omega()
    run_extended_profile(args.profile, period)


if __name__ == "__main__":
    main()
