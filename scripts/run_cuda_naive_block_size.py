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
BIN = ROOT / "bin" / "nbody_cuda_naive"
RESULTS = ROOT / "results"
PROFILES = {
    "smoke": {"N": [1024], "block_sizes": [64, 128], "steps": 2,
              "repeats": 2, "timeout": 180},
    "final": {"N": [4096, 8192, 16384],
              "block_sizes": [32, 64, 128, 256, 512], "steps": 20,
              "repeats": 7, "timeout": 1800},
}
DT = 5e-5
EPSILON = 0.01
SEED = 12345
REFERENCE_BLOCK_SIZE = 128
T_CRITICAL_95 = {1: 12.706205, 6: 2.446912}
RAW_FIELDS = [
    "sequence", "profile", "N", "block_size", "replica", "order_position",
    "steps", "force_evaluations", "pair_iterations", "compute_runtime_s",
    "backend_runtime_s", "command", "exit_code",
]
SUMMARY_FIELDS = [
    "profile", "N", "block_size", "repeats", "mean_compute_runtime_s",
    "median_compute_runtime_s", "std_compute_runtime_s", "sem_compute_runtime_s",
    "cv_compute_runtime_percent", "min_compute_runtime_s", "max_compute_runtime_s",
    "normalized_mean_runtime", "mean_paired_ratio_vs_b128",
    "median_paired_ratio_vs_b128", "std_paired_ratio_vs_b128",
    "sem_paired_ratio_vs_b128", "paired_ratio_ci95_low",
    "paired_ratio_ci95_high",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA naive block sizes with paired replicas."
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
    prefix = f"cuda_naive_block_size_{tag}"
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


def benchmark_command(n, block_size, steps):
    return [
        str(BIN), "--benchmark", "--N", str(n), "--steps", str(steps),
        "--dt", "5e-5", "--epsilon", str(EPSILON), "--init", "random",
        "--seed", str(SEED), "--integrator", "verlet",
        "--block-size", str(block_size),
    ]


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


def run_once(n, block_size, steps, timeout, phase):
    command = benchmark_command(n, block_size, steps)
    print(f"[{phase}] N={n} B={block_size}: {shlex.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=timeout
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


def balanced_order(block_sizes, replica, n_index):
    values = list(block_sizes)
    offset = (replica + n_index) % len(values)
    ordered = values[offset:] + values[:offset]
    return list(reversed(ordered)) if replica % 2 else ordered


def sample_statistics(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values))
    return mean, statistics.median(values), std, sem


def summarize(rows, profile):
    output = []
    grouped = {(n, block): [row for row in rows
                            if row["N"] == n and row["block_size"] == block]
               for n in profile["N"] for block in profile["block_sizes"]}
    for n in profile["N"]:
        minimum = min(statistics.mean(row["compute_runtime_s"] for row in grouped[(n, block)])
                      for block in profile["block_sizes"])
        reference = {row["replica"]: row for row in grouped[(n, REFERENCE_BLOCK_SIZE)]}
        for block_size in profile["block_sizes"]:
            selected = grouped[(n, block_size)]
            if len(selected) != profile["repeats"] or len({row["replica"] for row in selected}) != profile["repeats"]:
                raise RuntimeError(f"incomplete replicas for N={n}, B={block_size}")
            values = [row["compute_runtime_s"] for row in selected]
            mean, median, std, sem = sample_statistics(values)
            ratios = [row["compute_runtime_s"] /
                      reference[row["replica"]]["compute_runtime_s"] for row in selected]
            ratio_mean, ratio_median, ratio_std, ratio_sem = sample_statistics(ratios)
            critical = T_CRITICAL_95[len(ratios) - 1]
            output.append({
                "profile": profile["name"], "N": n, "block_size": block_size,
                "repeats": len(selected), "mean_compute_runtime_s": mean,
                "median_compute_runtime_s": median, "std_compute_runtime_s": std,
                "sem_compute_runtime_s": sem,
                "cv_compute_runtime_percent": 100.0 * std / mean,
                "min_compute_runtime_s": min(values), "max_compute_runtime_s": max(values),
                "normalized_mean_runtime": mean / minimum,
                "mean_paired_ratio_vs_b128": ratio_mean,
                "median_paired_ratio_vs_b128": ratio_median,
                "std_paired_ratio_vs_b128": ratio_std,
                "sem_paired_ratio_vs_b128": ratio_sem,
                "paired_ratio_ci95_low": ratio_mean - critical * ratio_sem,
                "paired_ratio_ci95_high": ratio_mean + critical * ratio_sem,
            })
    return output


def collect_metadata(args, profile, paths):
    gpu, driver = gpu_metadata()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node() or "unavailable",
        "gpu": gpu, "driver": driver,
        "cpu": file_value(Path("/proc/cpuinfo"), "model name"),
        "nvcc": command_output(["nvcc", "--version"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": command_output(["git", "status", "--porcelain"])
        not in ("", "unavailable"),
        "profile": args.profile, "backend": "cuda_naive",
        "N": profile["N"], "block_sizes": profile["block_sizes"],
        "reference_block_size": REFERENCE_BLOCK_SIZE,
        "steps": profile["steps"], "repeats": profile["repeats"],
        "dt": DT, "epsilon": EPSILON, "initializer": "random",
        "seed": SEED, "integrator": "verlet",
        "warm_up": "one unrecorded fresh process per N/block-size configuration",
        "order": "replica-major rotated and alternating block-size order",
        "fresh_process_per_replica": True,
        "outputs": {key: str(value) for key, value in paths.items()},
        "commands": [], "status": "running",
    }


def main():
    args = parse_args()
    os.chdir(ROOT)
    if not BIN.exists():
        raise SystemExit("Missing bin/nbody_cuda_naive; run `make cuda-naive CUDA_ARCH=sm_61`.")
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
            for block_size in balanced_order(profile["block_sizes"], 0, n_index):
                run_once(n, block_size, profile["steps"], profile["timeout"], "warm-up")
        for replica in range(profile["repeats"]):
            for n_index, n in enumerate(profile["N"]):
                for position, block_size in enumerate(
                        balanced_order(profile["block_sizes"], replica, n_index), 1):
                    (compute, backend), command, exit_code = run_once(
                        n, block_size, profile["steps"], profile["timeout"],
                        f"replica {replica + 1}/{profile['repeats']}",
                    )
                    sequence += 1
                    rows.append({
                        "sequence": sequence, "profile": args.profile, "N": n,
                        "block_size": block_size, "replica": replica + 1,
                        "order_position": position, "steps": profile["steps"],
                        "force_evaluations": profile["steps"] + 1,
                        "pair_iterations": (profile["steps"] + 1) * n * n,
                        "compute_runtime_s": compute, "backend_runtime_s": backend,
                        "command": shlex.join(command), "exit_code": exit_code,
                    })
                    write_csv(paths["raw_csv"], RAW_FIELDS, rows)
        summary = summarize(rows, profile)
        write_csv(paths["summary_csv"], SUMMARY_FIELDS, summary)
        metadata.update({"status": "completed",
                         "commands": [row["command"] for row in rows]})
        write_json(paths["metadata_json"], metadata)
        print("N,block_size,mean_compute_runtime_s,normalized_mean_runtime")
        for row in summary:
            print(f"{row['N']},{row['block_size']},{row['mean_compute_runtime_s']:.9g},"
                  f"{row['normalized_mean_runtime']:.9g}")
        for path in paths.values():
            print(path)
    except Exception as error:
        metadata.update({"status": "failed", "error": str(error),
                         "commands": [row["command"] for row in rows]})
        write_json(paths["metadata_json"], metadata)
        raise


if __name__ == "__main__":
    main()
