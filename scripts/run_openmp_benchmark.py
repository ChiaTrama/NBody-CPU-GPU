#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DT = 5e-5
DT_ARGUMENT = "5e-5"
EPSILON = 0.01
SEED = 12345
DEFAULT_THREADS = [1, 2, 4, 8]
BACKENDS = {
    "serial": ROOT / "bin" / "nbody_serial",
    "openmp": ROOT / "bin" / "nbody_openmp",
    "openmp_simd": ROOT / "bin" / "nbody_openmp_simd",
}
PROFILES = {
    "smoke": {"n_values": [64, 128], "steps": 2, "repeats": 2, "timeout": 30},
    "quick": {"n_values": [128, 256, 512, 1024, 2048, 4096], "steps": 20,
              "repeats": 3, "timeout": 300},
    "final": {"n_values": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
              "steps": 20, "repeats": 7, "timeout": 1800},
}
RAW_FIELDS = [
    "sequence", "replica", "backend", "N", "threads", "steps",
    "force_evaluations", "pair_iterations", "physical_nonself_pairs",
    "compute_runtime_s", "backend_runtime_s", "effective_interactions_per_s",
    "compute_speedup_vs_serial", "compute_parallel_efficiency", "command",
]
SUMMARY_FIELDS = [
    "backend", "N", "threads", "steps", "force_evaluations", "pair_iterations",
    "physical_nonself_pairs", "repeats",
    "mean_compute_runtime_s", "std_compute_runtime_s", "median_compute_runtime_s",
    "cv_compute_runtime_s", "mean_backend_runtime_s", "std_backend_runtime_s",
    "median_backend_runtime_s", "cv_backend_runtime_s",
    "mean_effective_interactions_per_s", "std_effective_interactions_per_s",
    "median_effective_interactions_per_s", "cv_effective_interactions_per_s",
    "mean_compute_speedup_vs_serial", "std_compute_speedup_vs_serial",
    "median_compute_speedup_vs_serial", "cv_compute_speedup_vs_serial",
    "mean_compute_parallel_efficiency", "std_compute_parallel_efficiency",
    "median_compute_parallel_efficiency", "cv_compute_parallel_efficiency",
]


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark serial, OpenMP, and experimental OpenMP-SIMD."
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--N", dest="n_values", nargs="+", type=positive_int)
    parser.add_argument("--threads", nargs="+", type=positive_int)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument("--timeout", type=positive_int)
    parser.add_argument("--output-tag", help="output suffix; defaults to profile")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    defaults = PROFILES[args.profile]
    args.n_values = sorted(set(args.n_values or defaults["n_values"]))
    args.threads = sorted(set(args.threads or DEFAULT_THREADS))
    args.steps = args.steps or defaults["steps"]
    args.repeats = args.repeats or defaults["repeats"]
    args.timeout = args.timeout or defaults["timeout"]
    args.output_tag = args.output_tag or args.profile
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_tag):
        parser.error("--output-tag may contain only letters, digits, '.', '_' and '-'")
    return args


def command_output(command):
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else "unavailable"


def file_value(path, prefix):
    try:
        for line in path.read_text().splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unavailable"


def output_paths(tag):
    return {
        "raw_csv": RESULTS_DIR / f"openmp_benchmark_raw_{tag}.csv",
        "summary_csv": RESULTS_DIR / f"openmp_benchmark_summary_{tag}.csv",
        "metadata_json": RESULTS_DIR / f"openmp_benchmark_metadata_{tag}.json",
    }


def prepare_outputs(paths, overwrite):
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        listed = "\n  ".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing output(s):\n  {listed}\nUse --overwrite.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def collect_metadata(args, paths):
    git_status = command_output(["git", "status", "--porcelain"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node() or "unavailable",
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_tag": command_output(["git", "tag", "--points-at", "HEAD"]),
        "git_dirty": git_status not in ("", "unavailable"),
        "cpu": file_value(Path("/proc/cpuinfo"), "model name"),
        "gpu": "not_applicable", "driver": "not_applicable",
        "gcc": command_output(["g++", "--version"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "build_commands": ["make"],
        "build_flags": {"CXXFLAGS": "-std=c++17 -O3 -Wall -Wextra -pedantic",
                        "OMPFLAGS": "-fopenmp"},
        "profile": args.profile, "backends": list(BACKENDS),
        "integrator": "verlet", "N": args.n_values, "steps": args.steps,
        "dt": DT, "epsilon": EPSILON, "initializer": "random", "seed": SEED,
        "threads": args.threads, "block_size": "not_applicable",
        "affinity": {"OMP_DYNAMIC": "false", "OMP_PROC_BIND": "close",
                     "OMP_PLACES": "cores"},
        "warm_up": "one unrecorded fresh process per backend/N/thread configuration",
        "repeats": args.repeats,
        "order": "replica-major alternating reverse with cyclic rotation",
        "timers": {"primary": "compute_runtime_s", "secondary": "backend_runtime_s"},
        "fresh_process_per_replica": True,
        "pair_iteration_convention": "force_evaluations * N^2",
        "outputs": {key: str(path) for key, path in paths.items()},
        "status": "initialized",
    }


def write_json(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_binaries():
    missing = [path for path in BACKENDS.values() if not path.exists()]
    if missing:
        listed = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"Missing required executable(s):\n  {listed}\nRun `make`.")


def benchmark_command(backend, n, steps):
    return [
        str(BACKENDS[backend]), "--benchmark", "--N", str(n), "--steps", str(steps),
        "--dt", DT_ARGUMENT, "--epsilon", str(EPSILON), "--init", "random",
        "--seed", str(SEED), "--integrator", "verlet",
    ]


def parse_benchmark_output(stdout, expected_steps):
    values = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition(",")
        if separator and key in {"compute_runtime_s", "backend_runtime_s",
                                 "benchmark_steps"}:
            values[key] = value
    missing = {"compute_runtime_s", "backend_runtime_s", "benchmark_steps"} - values.keys()
    if missing:
        raise RuntimeError(f"benchmark output missing fields: {sorted(missing)}")
    if int(values["benchmark_steps"]) != expected_steps:
        raise RuntimeError(
            f"benchmark_steps mismatch: expected {expected_steps}, "
            f"got {values['benchmark_steps']}"
        )
    compute = float(values["compute_runtime_s"])
    backend = float(values["backend_runtime_s"])
    if not all(math.isfinite(value) and value > 0.0 for value in (compute, backend)):
        raise RuntimeError("benchmark runtimes must be finite and positive")
    return compute, backend


def run_once(backend, n, threads, steps, timeout, phase):
    command = benchmark_command(backend, n, steps)
    env = os.environ.copy()
    if backend != "serial":
        env.update({"OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "false",
                    "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
    print(f"[{phase}] backend={backend} N={n} threads={threads}: "
          f"{shlex.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"benchmark timed out after {timeout}s: {shlex.join(command)}"
        ) from error
    if completed.returncode != 0:
        message = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            f"benchmark exited with {completed.returncode}: {shlex.join(command)}\n{message}"
        )
    return parse_benchmark_output(completed.stdout, steps), command


def balanced_order(values, replica):
    values = list(values)
    offset = (replica // 2) % len(values)
    ordered = values[offset:] + values[:offset]
    return list(reversed(ordered)) if replica % 2 else ordered


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def stats(values):
    if not values:
        return {"mean": "", "std": "", "median": "", "cv": ""}
    mean = statistics.mean(values)
    std = sample_std(values)
    return {"mean": mean, "std": std, "median": statistics.median(values),
            "cv": std / mean if mean else 0.0}


def add_paired_metrics(raw_rows):
    serial = {(row["replica"], row["N"]): row for row in raw_rows
              if row["backend"] == "serial"}
    for row in raw_rows:
        reference = serial[(row["replica"], row["N"])]
        if row["backend"] == "serial":
            row["compute_speedup_vs_serial"] = 1.0
            row["compute_parallel_efficiency"] = 1.0
        else:
            row["compute_speedup_vs_serial"] = (
                reference["compute_runtime_s"] / row["compute_runtime_s"]
            )
            row["compute_parallel_efficiency"] = (
                row["compute_speedup_vs_serial"] / row["threads"]
            )


def summarize(raw_rows):
    rows = []
    metrics = (
        "compute_runtime_s", "backend_runtime_s", "effective_interactions_per_s",
        "compute_speedup_vs_serial", "compute_parallel_efficiency",
    )
    keys = sorted({(row["backend"], row["N"], row["threads"]) for row in raw_rows})
    for backend, n, threads in keys:
        selected = [row for row in raw_rows if row["backend"] == backend
                    and row["N"] == n and row["threads"] == threads]
        first = selected[0]
        row = {
            "backend": backend, "N": n, "threads": threads,
            "steps": first["steps"], "force_evaluations": first["force_evaluations"],
            "pair_iterations": first["pair_iterations"],
            "physical_nonself_pairs": first["physical_nonself_pairs"],
            "repeats": len(selected),
        }
        for metric in metrics:
            for statistic, value in stats([item[metric] for item in selected]).items():
                row[f"{statistic}_{metric}"] = value
        rows.append(row)
    return rows


def main():
    args = parse_args()
    os.chdir(ROOT)
    require_binaries()
    paths = output_paths(args.output_tag)
    prepare_outputs(paths, args.overwrite)
    metadata = collect_metadata(args, paths)
    write_json(paths["metadata_json"], metadata)
    configurations = [("serial", n, 1) for n in args.n_values]
    configurations.extend((backend, n, threads) for n in args.n_values
                          for backend in ("openmp", "openmp_simd")
                          for threads in args.threads)
    raw_rows = []
    sequence = 0
    try:
        for index, (backend, n, threads) in enumerate(configurations):
            run_once(backend, n, threads, args.steps, args.timeout,
                     f"warm-up {index + 1}/{len(configurations)}")
        for replica in range(args.repeats):
            for backend, n, threads in balanced_order(configurations, replica):
                (compute, backend_time), command = run_once(
                    backend, n, threads, args.steps, args.timeout,
                    f"replica {replica + 1}/{args.repeats}",
                )
                sequence += 1
                force_evaluations = args.steps + 1
                pair_iterations = force_evaluations * n * n
                raw_rows.append({
                    "sequence": sequence, "replica": replica + 1,
                    "backend": backend, "N": n, "threads": threads,
                    "steps": args.steps, "force_evaluations": force_evaluations,
                    "pair_iterations": pair_iterations,
                    "physical_nonself_pairs": force_evaluations * n * (n - 1),
                    "compute_runtime_s": compute, "backend_runtime_s": backend_time,
                    "effective_interactions_per_s": pair_iterations / compute,
                    "compute_speedup_vs_serial": "", "compute_parallel_efficiency": "",
                    "command": shlex.join(command),
                })
                write_csv(paths["raw_csv"], RAW_FIELDS, raw_rows)
        add_paired_metrics(raw_rows)
        write_csv(paths["raw_csv"], RAW_FIELDS, raw_rows)
        summary = summarize(raw_rows)
        write_csv(paths["summary_csv"], SUMMARY_FIELDS, summary)
        metadata["status"] = "completed"
        metadata["measured_configurations"] = len(configurations)
        metadata["measured_commands"] = [row["command"] for row in raw_rows]
        write_json(paths["metadata_json"], metadata)
        print("backend,N,threads,mean_compute_runtime_s,compute_speedup_vs_serial")
        for row in summary:
            print(f"{row['backend']},{row['N']},{row['threads']},"
                  f"{row['mean_compute_runtime_s']:.9g},"
                  f"{row['mean_compute_speedup_vs_serial']:.9g}")
        for path in paths.values():
            print(path)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        metadata["measured_commands"] = [row["command"] for row in raw_rows]
        write_json(paths["metadata_json"], metadata)
        raise


if __name__ == "__main__":
    main()
