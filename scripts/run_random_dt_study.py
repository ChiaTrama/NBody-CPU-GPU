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
SERIAL_BIN = ROOT / "bin" / "nbody_serial"
RESULTS_ROOT = ROOT / "results" / "validation" / "random_dt_study_v2"
EPSILON_DT_RESULTS_ROOT = (
    ROOT / "results" / "validation" / "random_epsilon_dt_study_v1"
)
FIGURES_DIR = ROOT / "figures"

N = 32
SEED = 12345
BOX_SIZE = 10.0
MIN_DISTANCE = 0.1
EPSILON = 0.01
G = 1.0
OUTLIER_RATIO_THRESHOLD = 3.0
DOMINANT_MASS_FRACTION_THRESHOLD = 0.5
TIME_TOLERANCE_FACTOR = 64.0
TRAJECTORY_FIELDS = (
    "step", "time", "particle", "x", "y", "z", "vx", "vy", "vz", "mass"
)
ENERGY_FIELDS = ("step", "time", "kinetic", "potential", "total", "relative_error")
PROFILES = {
    "smoke": {"duration": 0.02, "dt_values": (0.002, 0.001)},
    "final": {
        "duration": 1.0,
        "dt_values": (2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5),
    },
}
EPSILON_DT_PROFILES = {
    "smoke": {
        "duration": 0.02,
        "epsilon_values": (0.005, 0.02),
        "dt_values": (0.002, 0.001),
    },
    "final": {
        "duration": 1.0,
        "epsilon_values": (0.005, 0.01, 0.02),
        "dt_values": (1.0e-4, 5.0e-5, 2.5e-5),
    },
}

SUMMARY_FIELDS = [
    "profile",
    "N",
    "seed",
    "box_size",
    "min_distance",
    "epsilon",
    "G",
    "integrator",
    "dt",
    "steps",
    "simulated_time",
    "save_every",
    "trajectory_frames",
    "energy_samples",
    "rms_includes_initial",
    "initial_energy",
    "max_relative_energy_error",
    "rms_relative_energy_error",
    "final_relative_energy_error",
    "all_values_finite",
    "initial_mass_weighted_rms_radius",
    "minimum_mass_weighted_rms_radius",
    "maximum_mass_weighted_rms_radius",
    "final_mass_weighted_rms_radius",
    "minimum_sampled_separation",
    "maximum_particle_displacement",
    "rms_final_particle_displacement",
    "max_final_displacement_to_rms_final_displacement",
    "maximum_radius_to_rms_radius",
    "maximum_particle_mass_fraction",
    "outlier_ratio_threshold",
    "dominant_mass_fraction_threshold",
    "outlier_indicator",
    "relative_final_radius_difference_vs_finest",
    "relative_minimum_separation_difference_vs_finest",
]
EPSILON_DT_SUMMARY_FIELDS = SUMMARY_FIELDS + [
    "relative_maximum_displacement_difference_vs_finest",
    "temporal_reference_dt",
    "observed_max_energy_metric_order_from_previous",
    "force_ratio_at_minimum_sampled_separation",
    "force_reduction_at_minimum_sampled_separation",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Study collective timestep sensitivity for seeded random N=32."
    )
    parser.add_argument(
        "--study",
        choices=("dt", "epsilon-dt"),
        default="dt",
        help="select the existing dt study or the joint epsilon-dt study",
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    return parser.parse_args()


def require_binary():
    if not SERIAL_BIN.exists():
        raise RuntimeError(f"missing executable: {SERIAL_BIN}; run `make` first")


def dt_tag(dt):
    return f"{dt:.8g}".replace(".", "p").replace("-", "m")


def epsilon_tag(epsilon):
    return f"{epsilon:.8g}".replace(".", "p").replace("-", "m")


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
    initial_masses = None
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
    for step in expected_steps:
        frame = frames[step]
        if set(frame) != expected_ids:
            raise RuntimeError(
                f"step {step} in {path} does not contain all {particle_count} IDs"
            )
        masses = [frame[particle]["mass"] for particle in range(particle_count)]
        if initial_masses is None:
            initial_masses = masses
        elif masses != initial_masses:
            raise RuntimeError(f"particle masses changed at step {step} in {path}")
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


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def rms_without_initial(values):
    if len(values) < 2:
        raise RuntimeError("RMS requires at least one sample after the initial state")
    return rms(values[1:])


def collective_metrics(frames):
    initial = frames[0]
    masses = [initial[particle]["mass"] for particle in range(N)]
    total_mass = sum(masses)
    initial_positions = [
        [initial[particle][axis] for axis in ("x", "y", "z")]
        for particle in range(N)
    ]
    rms_radii = []
    minimum_separation = math.inf
    maximum_displacement = 0.0
    maximum_radius_ratio = 0.0

    for frame in frames:
        positions = [
            [frame[particle][axis] for axis in ("x", "y", "z")]
            for particle in range(N)
        ]
        center_of_mass = [
            sum(masses[p] * positions[p][axis] for p in range(N)) / total_mass
            for axis in range(3)
        ]
        radii = [
            vector_norm(
                [positions[p][axis] - center_of_mass[axis] for axis in range(3)]
            )
            for p in range(N)
        ]
        rms_radius = math.sqrt(
            sum(masses[p] * radii[p] * radii[p] for p in range(N)) / total_mass
        )
        rms_radii.append(rms_radius)
        if rms_radius > 0.0:
            maximum_radius_ratio = max(maximum_radius_ratio, max(radii) / rms_radius)

        for i in range(N):
            displacement = vector_norm(
                [positions[i][axis] - initial_positions[i][axis] for axis in range(3)]
            )
            maximum_displacement = max(maximum_displacement, displacement)
            for j in range(i):
                minimum_separation = min(
                    minimum_separation,
                    vector_norm(
                        [positions[i][axis] - positions[j][axis] for axis in range(3)]
                    ),
                )

    final_positions = [
        [frames[-1][particle][axis] for axis in ("x", "y", "z")]
        for particle in range(N)
    ]
    final_displacements = [
        vector_norm(
            [final_positions[p][axis] - initial_positions[p][axis] for axis in range(3)]
        )
        for p in range(N)
    ]
    rms_final_displacement = math.sqrt(
        sum(masses[p] * final_displacements[p] ** 2 for p in range(N)) / total_mass
    )
    displacement_ratio = (
        max(final_displacements) / rms_final_displacement
        if rms_final_displacement > 0.0 else 0.0
    )
    mass_fraction = max(masses) / total_mass
    outlier = int(
        mass_fraction > DOMINANT_MASS_FRACTION_THRESHOLD
        or displacement_ratio > OUTLIER_RATIO_THRESHOLD
        or maximum_radius_ratio > OUTLIER_RATIO_THRESHOLD
    )
    return {
        "initial_mass_weighted_rms_radius": rms_radii[0],
        "minimum_mass_weighted_rms_radius": min(rms_radii),
        "maximum_mass_weighted_rms_radius": max(rms_radii),
        "final_mass_weighted_rms_radius": rms_radii[-1],
        "minimum_sampled_separation": minimum_separation,
        "maximum_particle_displacement": maximum_displacement,
        "rms_final_particle_displacement": rms_final_displacement,
        "max_final_displacement_to_rms_final_displacement": displacement_ratio,
        "maximum_radius_to_rms_radius": maximum_radius_ratio,
        "maximum_particle_mass_fraction": mass_fraction,
        "outlier_ratio_threshold": OUTLIER_RATIO_THRESHOLD,
        "dominant_mass_fraction_threshold": DOMINANT_MASS_FRACTION_THRESHOLD,
        "outlier_indicator": outlier,
    }


def simulator_command(epsilon, steps, dt, save_every, trajectory, energy):
    return [
        str(SERIAL_BIN),
        "--N", str(N),
        "--steps", str(steps),
        "--dt", f"{dt:.17g}",
        "--epsilon", f"{epsilon:.17g}",
        "--G", f"{G:.17g}",
        "--box-size", f"{BOX_SIZE:.17g}",
        "--min-distance", f"{MIN_DISTANCE:.17g}",
        "--init", "random",
        "--seed", str(SEED),
        "--integrator", "verlet",
        "--save-every", str(save_every),
        "--output", str(trajectory),
        "--energy-output", str(energy),
    ]


def run_case(study, profile, duration, epsilon, dt, runs_dir):
    steps = round(duration / dt)
    simulated_time = steps * dt
    if not math.isclose(simulated_time, duration, rel_tol=0.0, abs_tol=1.0e-14):
        raise RuntimeError(f"dt={dt} cannot represent requested duration {duration}")
    save_every = max(1, steps // 200)
    stem = f"random_N32_dt_{dt_tag(dt)}"
    if study == "epsilon-dt":
        stem = f"random_N32_epsilon_{epsilon_tag(epsilon)}_dt_{dt_tag(dt)}"
    trajectory = runs_dir / f"{stem}_trajectory.csv"
    energy = runs_dir / f"{stem}_energy.csv"
    command = simulator_command(epsilon, steps, dt, save_every, trajectory, energy)
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout, end="", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"epsilon={epsilon:g}, dt={dt:g} exited with {completed.returncode}"
        )

    frames = read_frames(trajectory, N, steps, save_every, dt)
    energy_rows = read_energy(energy, steps, save_every, dt)
    validate_matching_outputs(frames, energy_rows, trajectory)
    errors = [row["relative_error"] for row in energy_rows]
    row = {
        "profile": profile,
        "N": N,
        "seed": SEED,
        "box_size": BOX_SIZE,
        "min_distance": MIN_DISTANCE,
        "epsilon": epsilon,
        "G": G,
        "integrator": "verlet",
        "dt": dt,
        "steps": steps,
        "simulated_time": simulated_time,
        "save_every": save_every,
        "trajectory_frames": len(frames),
        "energy_samples": len(energy_rows),
        "rms_includes_initial": 0,
        "initial_energy": energy_rows[0]["total"],
        "max_relative_energy_error": max(errors),
        "rms_relative_energy_error": rms_without_initial(errors),
        "final_relative_energy_error": errors[-1],
        "all_values_finite": 1,
        "relative_final_radius_difference_vs_finest": "",
        "relative_minimum_separation_difference_vs_finest": "",
    }
    row.update(collective_metrics(frames))
    validate_summary_row(row)
    return row, initial_state(frames)


def initial_state(frames):
    initial = frames[0]
    return tuple(
        (
            particle,
            initial[particle]["mass"],
            *(initial[particle][axis] for axis in ("x", "y", "z")),
            *(initial[particle][axis] for axis in ("vx", "vy", "vz")),
        )
        for particle in range(N)
    )


def validate_same_initial_state(reference, candidate, label):
    if candidate != reference:
        raise RuntimeError(f"initial state mismatch for {label}")


def validate_summary_row(row):
    for key, value in row.items():
        if isinstance(value, str):
            continue
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite random-study metric {key}: {value}")


def add_collective_differences(rows):
    comparisons = (
        ("final_mass_weighted_rms_radius", "relative_final_radius_difference_vs_finest"),
        ("minimum_sampled_separation", "relative_minimum_separation_difference_vs_finest"),
    )
    for epsilon in sorted({row["epsilon"] for row in rows}):
        selected = [row for row in rows if row["epsilon"] == epsilon]
        finest = min(selected, key=lambda row: row["dt"])
        for row in selected:
            for source, destination in comparisons:
                scale = max(abs(finest[source]), sys.float_info.epsilon)
                row[destination] = abs(row[source] - finest[source]) / scale


def add_epsilon_dt_diagnostics(rows):
    add_collective_differences(rows)
    for epsilon in sorted({row["epsilon"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["epsilon"] == epsilon),
            key=lambda row: row["dt"],
            reverse=True,
        )
        finest = selected[-1]
        for row in selected:
            scale = max(
                abs(finest["maximum_particle_displacement"]),
                sys.float_info.epsilon,
            )
            row["relative_maximum_displacement_difference_vs_finest"] = abs(
                row["maximum_particle_displacement"]
                - finest["maximum_particle_displacement"]
            ) / scale
            row["temporal_reference_dt"] = finest["dt"]
            row["observed_max_energy_metric_order_from_previous"] = ""
            separation = row["minimum_sampled_separation"]
            ratio = separation ** 3 / (
                separation * separation + epsilon * epsilon
            ) ** 1.5
            row["force_ratio_at_minimum_sampled_separation"] = ratio
            row["force_reduction_at_minimum_sampled_separation"] = 1.0 - ratio

        for coarse, fine in zip(selected, selected[1:]):
            coarse_error = coarse["max_relative_energy_error"]
            fine_error = fine["max_relative_energy_error"]
            if min(coarse_error, fine_error) > 100.0 * sys.float_info.epsilon:
                fine["observed_max_energy_metric_order_from_previous"] = math.log(
                    coarse_error / fine_error
                ) / math.log(coarse["dt"] / fine["dt"])
        for row in selected:
            validate_summary_row(row)


def write_summary(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_epsilon_dt_summary(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPSILON_DT_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def scientific_tick(value, _):
    exponent = math.floor(math.log10(value))
    mantissa = value / 10.0 ** exponent
    return rf"${mantissa:g}\times10^{{{exponent}}}$"


def plot_summary(path, rows, profile, duration):
    selected = sorted(rows, key=lambda row: row["dt"])
    dt_values = [row["dt"] for row in selected]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for key, label in (
        ("max_relative_energy_error", "Maximum"),
        ("rms_relative_energy_error", "RMS"),
        ("final_relative_energy_error", "Final"),
    ):
        ax.loglog(dt_values, [row[key] for row in selected], marker="o", label=label)
    ax.axvline(5.0e-5, color="gray", linestyle="--", linewidth=1)
    ax.text(
        5.0e-5, 0.97, r"Chosen $\Delta t = 5\times10^{-5}$",
        transform=ax.get_xaxis_transform(), rotation=90,
        va="top", ha="right", color="dimgray",
    )
    ax.set_xlabel(r"Time step $\Delta t$")
    ax.set_xticks(dt_values)
    ax.xaxis.set_major_formatter(FuncFormatter(scientific_tick))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Relative energy error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(
        "Timestep sensitivity for a random N-body system\n"
        rf"$N = 32$, Velocity Verlet, $\epsilon = 0.01$, $T = {duration:g}$"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_epsilon_dt_summary(path, rows, profile, duration):
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8), constrained_layout=True
    )
    epsilon_values = sorted({row["epsilon"] for row in rows})
    for epsilon in epsilon_values:
        selected = sorted(
            (row for row in rows if row["epsilon"] == epsilon),
            key=lambda row: row["dt"],
        )
        axes[0].loglog(
            [row["dt"] for row in selected],
            [row["max_relative_energy_error"] for row in selected],
            marker="o",
            label=rf"$\epsilon = {epsilon:g}$",
        )
    dt_values = sorted({row["dt"] for row in rows})
    axes[0].set_title("Numerical stability")
    axes[0].set_xlabel(r"Time step $\Delta t$")
    axes[0].set_ylabel("Maximum relative energy error")
    axes[0].set_xticks(dt_values)
    axes[0].xaxis.set_major_formatter(FuncFormatter(scientific_tick))
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    finest_rows = [
        min((row for row in rows if row["epsilon"] == epsilon), key=lambda row: row["dt"])
        for epsilon in epsilon_values
    ]
    reductions = [
        100.0 * row["force_reduction_at_minimum_sampled_separation"]
        for row in finest_rows
    ]
    axes[1].plot(epsilon_values, reductions, marker="o")
    offsets = ((18, 8, "left"), (0, 10, "center"), (-8, -18, "right"))
    for row, reduction, (x_offset, y_offset, alignment) in zip(
            finest_rows, reductions, offsets):
        axes[1].annotate(
            rf"$r_{{min}} = {row['minimum_sampled_separation']:.3f}$",
            xy=(row["epsilon"], reduction), xytext=(x_offset, y_offset),
            textcoords="offset points", ha=alignment, fontsize=9,
        )
    axes[1].set_title("Softening at the closest sampled encounter")
    axes[1].set_xlabel(r"Softening length $\epsilon$")
    axes[1].set_ylabel(r"Force reduction at sampled $r_{min}$ [%]")
    axes[1].grid(True, alpha=0.3)
    axes[1].text(
        0.03, 0.97,
        r"Finest timestep: $\Delta t = 2.5\times10^{-5}$"
        "\n" r"$r_{min}$ evaluated over saved frames",
        transform=axes[1].transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.suptitle(
        "Softening and timestep sensitivity for a random N-body system\n"
        rf"$N = 32$, Velocity Verlet, $T = {duration:g}$"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.chdir(ROOT)
    require_binary()
    epsilon_dt_study = args.study == "epsilon-dt"
    config = (
        EPSILON_DT_PROFILES[args.profile]
        if epsilon_dt_study
        else PROFILES[args.profile]
    )
    results_root = EPSILON_DT_RESULTS_ROOT if epsilon_dt_study else RESULTS_ROOT
    output_dir = results_root / args.profile
    runs_dir = output_dir / "runs"
    study_name = "random_epsilon_dt_study" if epsilon_dt_study else "random_dt_study"
    summary = output_dir / f"{study_name}_{args.profile}.csv"
    plot_name = (
        f"{study_name}_{args.profile}.png"
        if epsilon_dt_study
        else f"random_dt_study_{args.profile}_v2.png"
    )
    plot = FIGURES_DIR / plot_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epsilon_values = config.get("epsilon_values", (EPSILON,))
    rows = []
    reference_initial_state = None
    for epsilon in epsilon_values:
        for dt in config["dt_values"]:
            row, state = run_case(
                args.study,
                args.profile,
                config["duration"],
                epsilon,
                dt,
                runs_dir,
            )
            if reference_initial_state is None:
                reference_initial_state = state
            else:
                validate_same_initial_state(
                    reference_initial_state,
                    state,
                    f"epsilon={epsilon:g}, dt={dt:g}",
                )
            if epsilon_dt_study and args.profile == "final":
                if row["trajectory_frames"] != 201 or row["energy_samples"] != 201:
                    raise RuntimeError(
                        f"epsilon={epsilon:g}, dt={dt:g} did not produce 201 frames"
                    )
            rows.append(row)

    if epsilon_dt_study:
        add_epsilon_dt_diagnostics(rows)
        write_epsilon_dt_summary(summary, rows)
        plot_epsilon_dt_summary(plot, rows, args.profile, config["duration"])
    else:
        add_collective_differences(rows)
        write_summary(summary, rows)
        plot_summary(plot, rows, args.profile, config["duration"])

    for row in rows:
        epsilon_label = (
            f"epsilon={row['epsilon']:.17g}," if epsilon_dt_study else ""
        )
        print(
            f"{epsilon_label}dt={row['dt']:.17g},"
            f"frames={row['trajectory_frames']},"
            f"max_energy_error={row['max_relative_energy_error']:.17g},"
            f"final_rms_radius={row['final_mass_weighted_rms_radius']:.17g},"
            f"min_sampled_separation={row['minimum_sampled_separation']:.17g},"
            f"outlier_indicator={row['outlier_indicator']}"
        )
    print("warning,minimum separation is measured only at saved frames")
    if epsilon_dt_study:
        print(
            "force_reduction_interpretation,evaluated at the minimum sampled "
            "separation; not a global simulation property"
        )
        print(
            "epsilon_interpretation,smaller epsilon stiffens close encounters; "
            "larger epsilon regularizes them while altering short-range forces"
        )
    print("interpretation,collective diagnostics do not require trajectory agreement")
    print(summary)
    print(plot)


if __name__ == "__main__":
    main()
