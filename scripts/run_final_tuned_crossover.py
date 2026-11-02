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
RESULTS = ROOT / "results"
BACKENDS = {
    "serial": ROOT / "bin" / "nbody_serial",
    "openmp_8": ROOT / "bin" / "nbody_openmp",
    "cuda_naive_b64": ROOT / "bin" / "nbody_cuda_naive",
    "cuda_tiled_b128": ROOT / "bin" / "nbody_cuda_tiled",
}
PROFILES = {
    "smoke": {"N": [1024], "steps": 2, "repeats": 2, "timeout": 180},
    "final": {"N": [512, 1024, 2048, 4096, 8192, 16384, 32768],
              "steps": 20, "repeats": 7, "timeout": 3600},
}
BLOCK_SIZES = {"serial": 0, "openmp_8": 0,
               "cuda_naive_b64": 64, "cuda_tiled_b128": 128}
THREADS = {"serial": 1, "openmp_8": 8,
           "cuda_naive_b64": 0, "cuda_tiled_b128": 0}
DT = 5e-5
EPSILON = 0.01
SEED = 12345
T_CRITICAL_95 = {1: 12.706205, 6: 2.446912}
RAW_FIELDS = [
    "sequence", "profile", "N", "backend", "threads", "block_size",
    "replica", "order_position", "steps", "force_evaluations",
    "pair_iterations", "compute_runtime_s", "backend_runtime_s",
    "backend_overhead_s", "pair_loop_iterations_per_s",
    "speedup_vs_serial", "command", "exit_code",
]
SUMMARY_FIELDS = [
    "profile", "N", "backend", "threads", "block_size", "count",
    "mean_compute_runtime_s", "median_compute_runtime_s",
    "std_compute_runtime_s", "sem_compute_runtime_s",
    "cv_compute_runtime_percent", "min_compute_runtime_s",
    "max_compute_runtime_s", "mean_backend_runtime_s",
    "mean_backend_overhead_s", "mean_pair_loop_iterations_per_s",
    "mean_paired_speedup_vs_serial", "median_paired_speedup_vs_serial",
    "std_paired_speedup_vs_serial", "sem_paired_speedup_vs_serial",
    "paired_speedup_ci95_low", "paired_speedup_ci95_high",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final paired serial/OpenMP/CUDA tuned crossover benchmark."
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--output-tag")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_tag = args.output_tag or f"v1_{args.profile}"
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


def gpu_metadata():
    value = command_output([
        "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"
    ])
    if value == "unavailable":
        return "unavailable", "unavailable"
    parts = [part.strip() for part in value.split(",", 1)]
    return (parts + ["unavailable"])[:2]


def output_paths(tag):
    prefix = f"final_tuned_crossover_{tag}"
    return {
        "raw_csv": RESULTS / f"{prefix}_raw.csv",
        "summary_csv": RESULTS / f"{prefix}_summary.csv",
        "metadata_json": RESULTS / f"{prefix}_metadata.json",
    }


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def benchmark_command(backend, n, steps):
    command = [
        str(BACKENDS[backend]), "--benchmark", "--N", str(n),
        "--steps", str(steps), "--dt", "5e-5", "--epsilon", str(EPSILON),
        "--init", "random", "--seed", str(SEED), "--integrator", "verlet",
    ]
    if BLOCK_SIZES[backend]:
        command.extend(["--block-size", str(BLOCK_SIZES[backend])])
    return command


def parse_times(stdout, expected_steps):
    values = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition(",")
        if separator and key in {
                "compute_runtime_s", "backend_runtime_s", "benchmark_steps"}:
            values[key] = value
    missing = {"compute_runtime_s", "backend_runtime_s", "benchmark_steps"} - values.keys()
    if missing:
        raise RuntimeError(f"benchmark output missing fields: {sorted(missing)}")
    if int(values["benchmark_steps"]) != expected_steps:
        raise RuntimeError("benchmark_steps does not match the requested value")
    compute, backend = float(values["compute_runtime_s"]), float(values["backend_runtime_s"])
    if not all(math.isfinite(value) and value > 0.0 for value in (compute, backend)):
        raise RuntimeError("benchmark runtimes must be finite and positive")
    if backend < compute:
        raise RuntimeError("backend runtime cannot be smaller than compute runtime")
    return compute, backend


def run_once(backend, n, steps, timeout, phase):
    command = benchmark_command(backend, n, steps)
    env = os.environ.copy()
    if backend == "openmp_8":
        env.update({"OMP_NUM_THREADS": "8", "OMP_DYNAMIC": "false"})
    print(f"[{phase}] backend={backend} N={n}: {shlex.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"benchmark timed out after {timeout}s: {shlex.join(command)}"
        ) from error
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            f"benchmark exited with {completed.returncode}: "
            f"{shlex.join(command)}\n{output}"
        )
    return parse_times(completed.stdout, steps), command, completed.returncode


def balanced_order(replica, n_index):
    values = list(BACKENDS)
    offset = (replica + n_index) % len(values)
    ordered = values[offset:] + values[:offset]
    return list(reversed(ordered)) if replica % 2 else ordered


def sample_statistics(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, statistics.median(values), std, std / math.sqrt(len(values))


def add_paired_speedups(rows):
    indexed = {(row["replica"], row["N"], row["backend"]): row for row in rows}
    for row in rows:
        serial = indexed[(row["replica"], row["N"], "serial")]
        row["speedup_vs_serial"] = (
            1.0 if row["backend"] == "serial"
            else serial["compute_runtime_s"] / row["compute_runtime_s"]
        )


def summarize(rows, profile):
    output = []
    for n in profile["N"]:
        for backend in BACKENDS:
            selected = [row for row in rows if row["N"] == n and row["backend"] == backend]
            if len(selected) != profile["repeats"] or len({row["replica"] for row in selected}) != profile["repeats"]:
                raise RuntimeError(f"incomplete replicas for N={n}, backend={backend}")
            compute = [row["compute_runtime_s"] for row in selected]
            backend_times = [row["backend_runtime_s"] for row in selected]
            overhead = [row["backend_overhead_s"] for row in selected]
            throughput = [row["pair_loop_iterations_per_s"] for row in selected]
            speedups = [row["speedup_vs_serial"] for row in selected]
            mean, median, std, sem = sample_statistics(compute)
            speed_mean, speed_median, speed_std, speed_sem = sample_statistics(speedups)
            critical = T_CRITICAL_95[len(speedups) - 1]
            output.append({
                "profile": profile["name"], "N": n, "backend": backend,
                "threads": THREADS[backend], "block_size": BLOCK_SIZES[backend],
                "count": len(selected), "mean_compute_runtime_s": mean,
                "median_compute_runtime_s": median, "std_compute_runtime_s": std,
                "sem_compute_runtime_s": sem,
                "cv_compute_runtime_percent": 100.0 * std / mean,
                "min_compute_runtime_s": min(compute),
                "max_compute_runtime_s": max(compute),
                "mean_backend_runtime_s": statistics.mean(backend_times),
                "mean_backend_overhead_s": statistics.mean(overhead),
                "mean_pair_loop_iterations_per_s": statistics.mean(throughput),
                "mean_paired_speedup_vs_serial": speed_mean,
                "median_paired_speedup_vs_serial": speed_median,
                "std_paired_speedup_vs_serial": speed_std,
                "sem_paired_speedup_vs_serial": speed_sem,
                "paired_speedup_ci95_low": speed_mean - critical * speed_sem,
                "paired_speedup_ci95_high": speed_mean + critical * speed_sem,
            })
    return output


def collect_metadata(args, profile, paths):
    gpu, driver = gpu_metadata()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node() or "unavailable",
        "cpu": file_value(Path("/proc/cpuinfo"), "model name"),
        "gpu": gpu, "driver": driver,
        "gcc": command_output(["g++", "--version"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": command_output(["git", "status", "--porcelain"])
        not in ("", "unavailable"),
        "profile": args.profile, "backends": list(BACKENDS),
        "threads": THREADS, "block_sizes": BLOCK_SIZES,
        "naive_block_size_rationale": (
            "B=64 had the best observed geometric normalized mean in tuning v1; "
            "paired B=64/B=128 confidence intervals included 1."
        ),
        "N": profile["N"], "steps": profile["steps"],
        "repeats": profile["repeats"], "dt": DT, "epsilon": EPSILON,
        "initializer": "random", "seed": SEED, "integrator": "verlet",
        "warm_up": "one unrecorded fresh process per backend/N configuration",
        "order": "replica-major rotated and alternating backend order",
        "affinity": {"OMP_NUM_THREADS": 8, "OMP_DYNAMIC": "false"},
        "fresh_process_per_replica": True,
        "pair_iteration_convention": "force_evaluations * N^2",
        "speedup_convention": "paired serial/backend for matching N and replica",
        "outputs": {key: str(value) for key, value in paths.items()},
        "commands": [], "status": "running",
    }


def main():
    args = parse_args()
    os.chdir(ROOT)
    missing = sorted({path for path in BACKENDS.values() if not path.exists()})
    if missing:
        raise SystemExit("Missing executable(s): " + ", ".join(map(str, missing)))
    profile = dict(PROFILES[args.profile], name=args.profile)
    paths = output_paths(args.output_tag)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit("Refusing to overwrite: " + ", ".join(map(str, existing)))
    RESULTS.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(args, profile, paths)
    write_json(paths["metadata_json"], metadata)
    rows = []
    sequence = 0
    try:
        for n_index, n in enumerate(profile["N"]):
            for backend in balanced_order(0, n_index):
                run_once(backend, n, profile["steps"], profile["timeout"], "warm-up")
        for replica in range(profile["repeats"]):
            for n_index, n in enumerate(profile["N"]):
                for position, backend in enumerate(balanced_order(replica, n_index), 1):
                    (compute, backend_time), command, exit_code = run_once(
                        backend, n, profile["steps"], profile["timeout"],
                        f"replica {replica + 1}/{profile['repeats']}",
                    )
                    sequence += 1
                    force_evaluations = profile["steps"] + 1
                    pair_iterations = force_evaluations * n * n
                    rows.append({
                        "sequence": sequence, "profile": args.profile, "N": n,
                        "backend": backend, "threads": THREADS[backend],
                        "block_size": BLOCK_SIZES[backend], "replica": replica + 1,
                        "order_position": position, "steps": profile["steps"],
                        "force_evaluations": force_evaluations,
                        "pair_iterations": pair_iterations,
                        "compute_runtime_s": compute,
                        "backend_runtime_s": backend_time,
                        "backend_overhead_s": backend_time - compute,
                        "pair_loop_iterations_per_s": pair_iterations / compute,
                        "speedup_vs_serial": "", "command": shlex.join(command),
                        "exit_code": exit_code,
                    })
                    write_csv(paths["raw_csv"], RAW_FIELDS, rows)
        add_paired_speedups(rows)
        write_csv(paths["raw_csv"], RAW_FIELDS, rows)
        summary = summarize(rows, profile)
        write_csv(paths["summary_csv"], SUMMARY_FIELDS, summary)
        metadata.update({"status": "completed",
                         "commands": [row["command"] for row in rows]})
        write_json(paths["metadata_json"], metadata)
        print("N,backend,mean_compute_runtime_s,paired_speedup_vs_serial")
        for row in summary:
            print(f"{row['N']},{row['backend']},{row['mean_compute_runtime_s']:.9g},"
                  f"{row['mean_paired_speedup_vs_serial']:.9g}")
        for path in paths.values():
            print(path)
    except Exception as error:
        metadata.update({"status": "failed", "error": str(error),
                         "commands": [row["command"] for row in rows]})
        write_json(paths["metadata_json"], metadata)
        raise


if __name__ == "__main__":
    main()
