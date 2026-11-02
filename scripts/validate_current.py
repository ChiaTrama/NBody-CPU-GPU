#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SERIAL_BIN = ROOT / "bin" / "nbody_serial"
OPENMP_BIN = ROOT / "bin" / "nbody_openmp"
OPENMP_SIMD_BIN = ROOT / "bin" / "nbody_openmp_simd"
CUDA_NAIVE_BIN = ROOT / "bin" / "nbody_cuda_naive"
CUDA_TILED_BIN = ROOT / "bin" / "nbody_cuda_tiled"
RESULTS_DIR = ROOT / "results" / "validation"
SUMMARY_CSV = RESULTS_DIR / "current_validation.csv"

EPSILON = 0.01
STATE_ATOL = 1.0e-12
STATE_RTOL = 1.0e-12
RANDOM_BOX_SIZE = 1000.0
RANDOM_MIN_DISTANCE = 1.0

CASES = [
    {
        "test": "two_body_euler",
        "integrator": "euler",
        "N": 2,
        "steps": 628,
        "dt": 0.01,
        "init": "two-body",
        "seed": None,
    },
    {
        "test": "two_body_verlet",
        "integrator": "verlet",
        "N": 2,
        "steps": 628,
        "dt": 0.01,
        "init": "two-body",
        "seed": None,
    },
    {
        "test": "two_body_elliptic_verlet",
        "integrator": "verlet",
        "N": 2,
        "steps": 20,
        "dt": 0.001,
        "init": "two-body-elliptic",
        "seed": None,
        "G": 1.0,
        "epsilon": 0.0,
    },
    {
        "test": "random_euler",
        "integrator": "euler",
        "N": 32,
        "steps": 20,
        "dt": 0.001,
        "init": "random",
        "seed": 12345,
        "box_size": RANDOM_BOX_SIZE,
        "min_distance": RANDOM_MIN_DISTANCE,
    },
    {
        "test": "random_verlet",
        "integrator": "verlet",
        "N": 32,
        "steps": 20,
        "dt": 0.001,
        "init": "random",
        "seed": 12345,
        "box_size": RANDOM_BOX_SIZE,
        "min_distance": RANDOM_MIN_DISTANCE,
    },
]

CUDA_TILED_CASES = [
    {
        "test": "tiled_random_verlet_n35_b32",
        "integrator": "verlet",
        "N": 35,
        "steps": 2,
        "dt": 0.001,
        "init": "random",
        "seed": 12345,
        "block_size": 32,
    },
    {
        "test": "tiled_barycentric_verlet_n2_b64",
        "integrator": "verlet",
        "N": 2,
        "steps": 3,
        "dt": 0.001,
        "init": "two-body-barycentric",
        "seed": None,
        "block_size": 64,
    },
    {
        "test": "tiled_random_euler_n31_b64",
        "integrator": "euler",
        "N": 31,
        "steps": 3,
        "dt": 0.001,
        "init": "random",
        "seed": 12345,
        "block_size": 64,
    },
    {
        "test": "tiled_random_verlet_n33_b128",
        "integrator": "verlet",
        "N": 33,
        "steps": 3,
        "dt": 0.001,
        "init": "random",
        "seed": 12345,
        "block_size": 128,
    },
    {
        "test": "tiled_random_euler_n100_b64",
        "integrator": "euler",
        "N": 100,
        "steps": 2,
        "dt": 0.001,
        "init": "random",
        "seed": 12345,
        "block_size": 64,
    },
    {
        "test": "tiled_random_verlet_n257_b256",
        "integrator": "verlet",
        "N": 257,
        "steps": 2,
        "dt": 0.001,
        "init": "random",
        "seed": 12345,
        "block_size": 256,
    },
    {
        "test": "tiled_random_verlet_n513_b512",
        "integrator": "verlet",
        "N": 513,
        "steps": 3,
        "dt": 0.001,
        "init": "random",
        "seed": 12345,
        "block_size": 512,
    },
]

STATE_COLUMNS = ["x", "y", "z", "vx", "vy", "vz"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the current NBody quality gate.")
    parser.add_argument(
        "--include-cuda",
        action="store_true",
        help="Include CUDA naive and tiled validation. This fails if CUDA is unavailable.",
    )
    return parser.parse_args()


def backend_specs(include_cuda):
    specs = [
        {"backend": "serial", "binary": SERIAL_BIN, "threads": None},
        {"backend": "openmp", "binary": OPENMP_BIN, "threads": 4},
    ]

    if OPENMP_SIMD_BIN.exists():
        specs.append({"backend": "openmp_simd", "binary": OPENMP_SIMD_BIN, "threads": 4})

    if include_cuda:
        specs.extend([
            {"backend": "cuda_naive", "binary": CUDA_NAIVE_BIN, "threads": None},
            {"backend": "cuda_tiled", "binary": CUDA_TILED_BIN, "threads": None},
        ])

    return specs


def require_binaries(specs):
    missing = [spec["binary"] for spec in specs if not spec["binary"].exists()]
    if missing:
        print("Missing required executable(s):", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(
            "Run `make`, `make cuda-naive CUDA_ARCH=sm_61`, and "
            "`make cuda-tiled CUDA_ARCH=sm_61`.",
            file=sys.stderr,
        )
        sys.exit(1)


def simulator_command(binary, case, trajectory, energy):
    command = [
        str(binary),
        "--N", str(case["N"]),
        "--steps", str(case["steps"]),
        "--dt", str(case["dt"]),
        "--epsilon", str(case.get("epsilon", EPSILON)),
        "--init", case["init"],
        "--integrator", case["integrator"],
        "--save-every", str(case["steps"]),
        "--output", str(trajectory),
        "--energy-output", str(energy),
    ]

    if case["seed"] is not None:
        command.extend(["--seed", str(case["seed"])])
    if case.get("G") is not None:
        command.extend(["--G", str(case["G"])])
    if case.get("box_size") is not None:
        command.extend(["--box-size", str(case["box_size"])])
    if case.get("min_distance") is not None:
        command.extend(["--min-distance", str(case["min_distance"])])
    if case.get("block_size") is not None:
        command.extend(["--block-size", str(case["block_size"])])

    return command


def run_command(command, threads=None):
    env = os.environ.copy()
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_DYNAMIC"] = "false"

    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def parse_stdout_energies(stdout):
    values = {}

    for line in stdout.splitlines():
        if "," not in line:
            continue
        key, value = line.split(",", 1)
        if key in ("initial_energy", "final_energy", "relative_energy_error"):
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"non-finite stdout value {key}: {value}")
            values[key] = parsed

    missing = [
        key for key in ("initial_energy", "final_energy", "relative_energy_error")
        if key not in values
    ]
    if missing:
        raise ValueError(f"simulator stdout missing: {', '.join(missing)}")

    return values


def parse_numeric_csv(path):
    rows = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"non-finite CSV value in {path}: {key}={value}")
                parsed[key] = number
            rows.append(parsed)

    return rows


def final_state_from_trajectory(path, expected_step, expected_n):
    rows = parse_numeric_csv(path)
    initial_rows = [row for row in rows if int(row["step"]) == 0]
    final_rows = [row for row in rows if int(row["step"]) == expected_step]

    if len(initial_rows) != expected_n:
        raise ValueError(
            f"{path} does not contain step 0 for all particles "
            f"(found {len(initial_rows)}, expected {expected_n})"
        )
    if len(final_rows) != expected_n:
        raise ValueError(
            f"{path} does not contain final step {expected_step} for all particles "
            f"(found {len(final_rows)}, expected {expected_n})"
        )

    particle_values = [row["particle"] for row in initial_rows + final_rows]
    if any(not value.is_integer() for value in particle_values):
        raise ValueError(f"{path} contains a non-integer particle id at final step")

    initial_rows.sort(key=lambda row: int(row["particle"]))
    final_rows.sort(key=lambda row: int(row["particle"]))
    initial_ids = [int(row["particle"]) for row in initial_rows]
    particle_ids = [int(row["particle"]) for row in final_rows]
    expected_ids = list(range(expected_n))
    if initial_ids != expected_ids or particle_ids != expected_ids:
        raise ValueError(
            f"{path} particle ids are step 0={initial_ids}, final={particle_ids}; "
            f"expected 0 through {expected_n - 1}"
        )

    for initial, final in zip(initial_rows, final_rows):
        if final["mass"] != initial["mass"]:
            raise ValueError(
                f"{path} particle {int(initial['particle'])} mass changed from "
                f"{initial['mass']} to {final['mass']}"
            )

    return final_rows


def validate_random_trajectory(path, case):
    rows = parse_numeric_csv(path)
    box_size = case.get("box_size", RANDOM_BOX_SIZE)
    min_distance = case.get("min_distance", RANDOM_MIN_DISTANCE)
    required_columns = {
        "step", "particle", "x", "y", "z", "vx", "vy", "vz", "mass"
    }
    if not rows:
        raise ValueError(f"{path} contains no trajectory rows")

    missing_columns = required_columns - rows[0].keys()
    if missing_columns:
        raise ValueError(
            f"{path} is missing trajectory columns: {', '.join(sorted(missing_columns))}"
        )

    initial_rows = [row for row in rows if int(row["step"]) == 0]
    if len(initial_rows) != case["N"]:
        raise ValueError(
            f"{path} initial step has {len(initial_rows)} particles, "
            f"expected {case['N']}"
        )

    particle_values = [row["particle"] for row in initial_rows]
    if any(not value.is_integer() for value in particle_values):
        raise ValueError(f"{path} contains a non-integer particle id at step 0")

    initial_rows.sort(key=lambda row: int(row["particle"]))
    particle_ids = [int(row["particle"]) for row in initial_rows]
    if particle_ids != list(range(case["N"])):
        raise ValueError(
            f"{path} initial particle ids are {particle_ids}, "
            f"expected 0 through {case['N'] - 1}"
        )

    initial_mass = {}
    for row in initial_rows:
        particle = int(row["particle"])
        mass = row["mass"]
        if not 1.0 <= mass < 10.0:
            raise ValueError(f"{path} particle {particle} mass {mass} is outside [1, 10)")

        for column in ("x", "y", "z"):
            value = row[column]
            if not 0.0 <= value < box_size:
                raise ValueError(
                    f"{path} particle {particle} {column}={value} is outside "
                    f"[0, {box_size})"
                )

        for column in ("vx", "vy", "vz"):
            value = row[column]
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"{path} particle {particle} {column}={value} is outside [-1, 1]"
                )

        initial_mass[particle] = mass

    minimum_distance_squared = min_distance ** 2
    for i, first in enumerate(initial_rows):
        for second in initial_rows[:i]:
            dx = first["x"] - second["x"]
            dy = first["y"] - second["y"]
            dz = first["z"] - second["z"]
            distance_squared = dx * dx + dy * dy + dz * dz
            if distance_squared < minimum_distance_squared:
                raise ValueError(
                    f"{path} particles {int(first['particle'])} and "
                    f"{int(second['particle'])} violate min_distance={min_distance}"
                )

    for row in rows:
        particle_value = row["particle"]
        if not particle_value.is_integer():
            raise ValueError(f"{path} contains a non-integer particle id")
        particle = int(particle_value)
        if particle not in initial_mass:
            raise ValueError(f"{path} contains unexpected particle id {particle}")
        if row["mass"] != initial_mass[particle]:
            raise ValueError(
                f"{path} particle {particle} mass changed from "
                f"{initial_mass[particle]} to {row['mass']}"
            )


def validate_energy_csv(path, expected_step):
    rows = parse_numeric_csv(path)
    if not any(int(row["step"]) == expected_step for row in rows):
        raise ValueError(f"{path} does not contain final step {expected_step}")


def compare_states(reference, candidate):
    max_abs = 0.0
    max_normalized = 0.0
    failures = []

    for ref_row, cand_row in zip(reference, candidate):
        particle = int(ref_row["particle"])
        for column in STATE_COLUMNS:
            ref = ref_row[column]
            cand = cand_row[column]
            diff = abs(cand - ref)
            tolerance = STATE_ATOL + STATE_RTOL * abs(ref)
            normalized = diff / tolerance

            max_abs = max(max_abs, diff)
            max_normalized = max(max_normalized, normalized)

            if diff > tolerance:
                failures.append(
                    f"particle {particle} {column}: candidate={cand} "
                    f"reference={ref} diff={diff} tolerance={tolerance}"
                )

    return max_abs, max_normalized, failures


def append_failure(row, message):
    if row["status"] == "PASS":
        row["status"] = "FAIL"
    if row["details"]:
        row["details"] += "; "
    row["details"] += message


def run_validation_case(case, spec, references):
    stem = f"{case['test']}_{spec['backend']}"
    trajectory = RESULTS_DIR / f"{stem}.csv"
    energy = RESULTS_DIR / f"{stem}_energy.csv"

    for path in (trajectory, energy):
        if path.exists():
            path.unlink()

    command = simulator_command(spec["binary"], case, trajectory, energy)
    completed = run_command(command, spec["threads"])

    row = {
        "test": case["test"],
        "backend": spec["backend"],
        "integrator": case["integrator"],
        "N": case["N"],
        "steps": case["steps"],
        "relative_energy_error": "",
        "max_abs_state_diff": "",
        "max_normalized_state_diff": "",
        "status": "PASS",
        "details": "",
    }
    final_state = None

    if completed.returncode != 0:
        append_failure(
            row,
            f"return code {completed.returncode}; stderr={completed.stderr.strip()}",
        )
        return row, final_state

    try:
        energies = parse_stdout_energies(completed.stdout)
        row["relative_energy_error"] = energies["relative_energy_error"]
    except ValueError as error:
        append_failure(row, str(error))

    for path in (trajectory, energy):
        if not path.exists():
            append_failure(row, f"missing output file {path}")

    if trajectory.exists():
        try:
            final_state = final_state_from_trajectory(trajectory, case["steps"], case["N"])
            if case["init"] == "random":
                validate_random_trajectory(trajectory, case)
        except ValueError as error:
            append_failure(row, str(error))

    if energy.exists():
        try:
            validate_energy_csv(energy, case["steps"])
        except ValueError as error:
            append_failure(row, str(error))

    if spec["backend"] == "serial" and final_state is not None:
        references[case["test"]] = final_state
        row["max_abs_state_diff"] = 0.0
        row["max_normalized_state_diff"] = 0.0
    elif final_state is not None:
        reference = references.get(case["test"])
        if reference is None:
            append_failure(row, "missing serial reference state")
        else:
            max_abs, max_normalized, state_failures = compare_states(reference, final_state)
            row["max_abs_state_diff"] = max_abs
            row["max_normalized_state_diff"] = max_normalized
            for failure in state_failures:
                append_failure(row, failure)

    return row, final_state


def apply_physical_thresholds(rows):
    row_by_key = {(row["test"], row["backend"]): row for row in rows}

    for row in rows:
        if row["status"] != "PASS" or row["relative_energy_error"] == "":
            continue

        rel_error = float(row["relative_energy_error"])
        if row["test"] == "two_body_verlet" and rel_error >= 1.0e-8:
            append_failure(row, f"two-body Verlet relative energy error {rel_error} >= 1e-8")
        if row["test"] == "random_verlet" and rel_error >= 1.0e-10:
            append_failure(row, f"random Verlet relative energy error {rel_error} >= 1e-10")

    backends = sorted({row["backend"] for row in rows})
    for backend in backends:
        euler = row_by_key.get(("two_body_euler", backend))
        verlet = row_by_key.get(("two_body_verlet", backend))

        if (
            euler is None or
            verlet is None or
            euler["relative_energy_error"] == "" or
            verlet["relative_energy_error"] == ""
        ):
            continue

        euler_error = float(euler["relative_energy_error"])
        verlet_error = float(verlet["relative_energy_error"])
        if not euler_error > verlet_error:
            append_failure(
                euler,
                f"two-body Euler error {euler_error} is not greater than Verlet error {verlet_error}",
            )


def negative_cases(include_cuda):
    zero_energy_g = 0.5 * math.sqrt(1.0 + EPSILON * EPSILON)

    cases = [
        {
            "name": "invalid_integrator",
            "command": [
                str(SERIAL_BIN),
                "--N", "2",
                "--steps", "1",
                "--dt", "0.01",
                "--epsilon", str(EPSILON),
                "--init", "two-body",
                "--integrator", "not-an-integrator",
                "--output", str(RESULTS_DIR / "negative_invalid_integrator.csv"),
                "--energy-output", str(RESULTS_DIR / "negative_invalid_integrator_energy.csv"),
            ],
            "message": None,
        },
        {
            "name": "two_body_wrong_N",
            "command": [
                str(SERIAL_BIN),
                "--N", "3",
                "--steps", "1",
                "--dt", "0.01",
                "--epsilon", str(EPSILON),
                "--init", "two-body",
                "--integrator", "verlet",
                "--output", str(RESULTS_DIR / "negative_two_body_wrong_N.csv"),
                "--energy-output", str(RESULTS_DIR / "negative_two_body_wrong_N_energy.csv"),
            ],
            "message": None,
        },
        {
            "name": "two_body_elliptic_wrong_N",
            "command": [
                str(SERIAL_BIN),
                "--N", "3",
                "--steps", "1",
                "--dt", "0.001",
                "--epsilon", "0",
                "--G", "1",
                "--init", "two-body-elliptic",
                "--integrator", "verlet",
                "--output", str(RESULTS_DIR / "negative_two_body_elliptic_wrong_N.csv"),
                "--energy-output", str(
                    RESULTS_DIR / "negative_two_body_elliptic_wrong_N_energy.csv"
                ),
            ],
            "message": "--init two-body-elliptic requires --N 2",
        },
        {
            "name": "negative_dt",
            "command": [
                str(SERIAL_BIN),
                "--N", "2",
                "--steps", "1",
                "--dt", "-0.01",
                "--epsilon", str(EPSILON),
                "--init", "two-body",
                "--integrator", "verlet",
                "--output", str(RESULTS_DIR / "negative_dt.csv"),
                "--energy-output", str(RESULTS_DIR / "negative_dt_energy.csv"),
            ],
            "message": None,
        },
        {
            "name": "near_zero_initial_energy",
            "command": [
                str(SERIAL_BIN),
                "--N", "2",
                "--steps", "1",
                "--dt", "0.01",
                "--epsilon", str(EPSILON),
                "--G", f"{zero_energy_g:.17g}",
                "--init", "two-body",
                "--integrator", "verlet",
                "--save-every", "1",
                "--output", str(RESULTS_DIR / "negative_near_zero_energy.csv"),
                "--energy-output", str(RESULTS_DIR / "negative_near_zero_energy_energy.csv"),
            ],
            "message": "relative energy error is undefined",
        },
        {
            "name": "block_size_zero",
            "command": [str(SERIAL_BIN), "--block-size", "0"],
            "message": "invalid value for --block-size",
        },
        {
            "name": "block_size_negative",
            "command": [str(SERIAL_BIN), "--block-size", "-64"],
            "message": "invalid value for --block-size",
        },
        {
            "name": "block_size_non_numeric",
            "command": [str(SERIAL_BIN), "--block-size", "abc"],
            "message": "invalid value for --block-size",
        },
        {
            "name": "block_size_missing",
            "command": [str(SERIAL_BIN), "--block-size"],
            "message": "missing value for --block-size",
        },
    ]

    if include_cuda:
        cases.extend([
            {
                "name": "naive_block_size_exceeds_device",
                "command": [
                    str(CUDA_NAIVE_BIN),
                    "--benchmark", "--N", "2", "--steps", "1",
                    "--init", "two-body", "--integrator", "verlet",
                    "--block-size", "2147483647",
                ],
                "message": "exceeds device maxThreadsPerBlock",
            },
            {
                "name": "tiled_block_size_exceeds_device",
                "command": [
                    str(CUDA_TILED_BIN),
                    "--benchmark",
                    "--N", "2",
                    "--steps", "1",
                    "--init", "two-body",
                    "--integrator", "verlet",
                    "--block-size", "2147483647",
                ],
                "message": "exceeds device maxThreadsPerBlock",
            },
        ])

    return cases


def run_negative_tests(include_cuda):
    results = []

    for case in negative_cases(include_cuda):
        completed = run_command(case["command"])
        combined_output = completed.stdout + completed.stderr
        passed = completed.returncode != 0

        if case["message"] is not None and case["message"] not in combined_output:
            passed = False

        results.append(
            {
                "name": case["name"],
                "status": "PASS" if passed else "FAIL",
                "returncode": completed.returncode,
                "message": case["message"] or "",
                "output": combined_output.strip(),
            }
        )

    return results


def write_summary(rows):
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "test",
                "backend",
                "integrator",
                "N",
                "steps",
                "relative_energy_error",
                "max_abs_state_diff",
                "max_normalized_state_diff",
                "status",
                "details",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, negative_results):
    print(
        f"{'test':18} {'backend':12} {'integrator':10} {'N':>4} {'steps':>6} "
        f"{'rel_error':>14} {'max_abs':>14} {'max_norm':>14} {'status':>8}"
    )
    for row in rows:
        rel_error = row["relative_energy_error"]
        max_abs = row["max_abs_state_diff"]
        max_norm = row["max_normalized_state_diff"]

        rel_text = f"{float(rel_error):14.6g}" if rel_error != "" else f"{'':>14}"
        max_abs_text = f"{float(max_abs):14.6g}" if max_abs != "" else f"{'':>14}"
        max_norm_text = f"{float(max_norm):14.6g}" if max_norm != "" else f"{'':>14}"

        print(
            f"{row['test']:18} "
            f"{row['backend']:12} "
            f"{row['integrator']:10} "
            f"{row['N']:>4} "
            f"{row['steps']:>6} "
            f"{rel_text} "
            f"{max_abs_text} "
            f"{max_norm_text} "
            f"{row['status']:>8}"
        )

    print()
    print(f"{'negative_test':28} {'returncode':>10} {'status':>8}")
    for result in negative_results:
        print(f"{result['name']:28} {result['returncode']:>10} {result['status']:>8}")


def compare_cuda_states(rows, states):
    comparisons = []
    rows_by_key = {(row["test"], row["backend"]): row for row in rows}

    for test in sorted({test for test, backend in states if backend == "cuda_tiled"}):
        naive = states.get((test, "cuda_naive"))
        tiled = states.get((test, "cuda_tiled"))
        if naive is None or tiled is None:
            continue

        max_abs, max_normalized, failures = compare_states(naive, tiled)
        comparisons.append({
            "test": test,
            "max_abs": max_abs,
            "max_normalized": max_normalized,
            "status": "FAIL" if failures else "PASS",
        })

        tiled_row = rows_by_key[(test, "cuda_tiled")]
        for failure in failures:
            append_failure(tiled_row, f"vs CUDA naive: {failure}")

    return comparisons


def print_cuda_comparisons(comparisons):
    if not comparisons:
        return

    print()
    print(f"{'cuda_naive_vs_tiled':34} {'max_abs':>14} {'max_norm':>14} {'status':>8}")
    for comparison in comparisons:
        print(
            f"{comparison['test']:34} "
            f"{comparison['max_abs']:14.6g} "
            f"{comparison['max_normalized']:14.6g} "
            f"{comparison['status']:>8}"
        )


def collect_failures(rows, negative_results):
    failures = []

    for row in rows:
        if row["status"] != "PASS":
            failures.append(f"{row['test']} {row['backend']}: {row['details']}")

    for result in negative_results:
        if result["status"] != "PASS":
            failures.append(
                f"negative {result['name']}: returncode={result['returncode']} "
                f"expected_message={result['message']} output={result['output']}"
            )

    return failures


def main():
    args = parse_args()

    os.chdir(ROOT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = backend_specs(args.include_cuda)
    require_binaries(specs)

    rows = []
    references = {}
    states = {}

    for case in CASES:
        serial_spec = next(spec for spec in specs if spec["backend"] == "serial")
        row, final_state = run_validation_case(case, serial_spec, references)
        rows.append(row)
        states[(case["test"], "serial")] = final_state

        for spec in specs:
            if spec["backend"] == "serial":
                continue
            row, final_state = run_validation_case(case, spec, references)
            rows.append(row)
            states[(case["test"], spec["backend"])] = final_state

    if args.include_cuda:
        cuda_specs = [
            spec for spec in specs
            if spec["backend"] in ("serial", "cuda_naive", "cuda_tiled")
        ]
        for case in CUDA_TILED_CASES:
            for spec in cuda_specs:
                row, final_state = run_validation_case(case, spec, references)
                rows.append(row)
                states[(case["test"], spec["backend"])] = final_state

    apply_physical_thresholds(rows)
    cuda_comparisons = compare_cuda_states(rows, states)
    negative_results = run_negative_tests(args.include_cuda)
    write_summary(rows)
    print_table(rows, negative_results)
    print_cuda_comparisons(cuda_comparisons)

    failures = collect_failures(rows, negative_results)
    passed_count = len(rows) + len(negative_results) - len(failures)
    failed_count = len(failures)

    print(SUMMARY_CSV)
    print(f"validation_passed,{passed_count}")
    print(f"validation_failed,{failed_count}")

    if failures:
        print("Validation failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
