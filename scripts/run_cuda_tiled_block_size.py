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
import sys


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "nbody_cuda_tiled"
RESULTS_DIR = ROOT / "results"
DT = 5e-5
DT_ARGUMENT = "5e-5"
EPSILON = 0.01
SEED = 12345
DEFAULT_BLOCK_SIZES = [32, 64, 128, 256, 512]
PROFILES = {
    "smoke": {"n_values": [4096, 4097], "steps": 5, "repeats": 2, "timeout": 120},
    "quick": {"n_values": [4096, 8192, 10000], "steps": 20, "repeats": 3,
              "timeout": 900},
    "final": {"n_values": [4096, 8192, 10000, 16384], "steps": 20,
              "repeats": 7, "timeout": 1800},
}
RAW_FIELDS = [
    "sequence", "replica", "backend", "N", "block_size", "steps",
    "force_evaluations", "pair_iterations", "physical_nonself_pairs",
    "compute_runtime_s", "backend_runtime_s", "effective_interactions_per_s",
    "command",
]
SUMMARY_FIELDS = [
    "backend", "N", "block_size", "steps", "force_evaluations",
    "pair_iterations", "physical_nonself_pairs", "repeats",
    "mean_compute_runtime_s", "std_compute_runtime_s",
    "median_compute_runtime_s", "cv_compute_runtime",
    "mean_backend_runtime_s", "std_backend_runtime_s",
    "median_backend_runtime_s", "cv_backend_runtime",
    "mean_effective_interactions_per_s", "std_effective_interactions_per_s",
    "median_effective_interactions_per_s", "cv_effective_interactions_per_s",
    "normalized_compute_runtime", "ranking_position",
    "geometric_mean_normalized_runtime", "mean_compute_cv_across_n",
    "selection_status", "selection_reason",
]


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA tiled block sizes using internal timing."
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--N", dest="n_values", nargs="+", type=positive_int)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument("--timeout", type=positive_int)
    parser.add_argument("--block-sizes", nargs="+", type=positive_int)
    parser.add_argument("--output-tag", help="output suffix; defaults to profile")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    defaults = PROFILES[args.profile]
    args.n_values = sorted(set(args.n_values or defaults["n_values"]))
    args.steps = args.steps or defaults["steps"]
    args.repeats = args.repeats or defaults["repeats"]
    args.timeout = args.timeout or defaults["timeout"]
    args.block_sizes = sorted(set(args.block_sizes or DEFAULT_BLOCK_SIZES))
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


def gpu_metadata():
    query = [
        "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"
    ]
    value = command_output(query)
    if value == "unavailable":
        return "unavailable", "unavailable"
    parts = [part.strip() for part in value.split(",", 1)]
    return (parts + ["unavailable"])[:2]


def collect_metadata(args, paths):
    gpu, driver = gpu_metadata()
    status = command_output(["git", "status", "--porcelain"])
    dirty = status not in ("", "unavailable")
    tags = command_output(["git", "tag", "--points-at", "HEAD"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node() or "unavailable",
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_tag": tags,
        "git_dirty": dirty,
        "cpu": file_value(Path("/proc/cpuinfo"), "model name"),
        "gpu": gpu,
        "driver": driver,
        "gcc": command_output(["g++", "--version"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "build_commands": ["make", "make cuda-tiled CUDA_ARCH=sm_61"],
        "build_flags": {
            "CXXFLAGS": "-std=c++17 -O3 -Wall -Wextra -pedantic",
            "NVCCFLAGS": "-std=c++17 -O3 -arch=sm_61",
        },
        "profile": args.profile,
        "backend": "cuda_tiled",
        "integrator": "verlet",
        "N": args.n_values,
        "steps": args.steps,
        "dt": DT,
        "epsilon": EPSILON,
        "initializer": "random",
        "seed": SEED,
        "threads": 0,
        "affinity": "not_applicable",
        "block_sizes_requested": args.block_sizes,
        "warm_up": "one unrecorded fresh process per configuration",
        "repeats": args.repeats,
        "order": "replica-major alternating reverse with cyclic rotation",
        "timer": {
            "primary": "compute_runtime_s from CUDA events",
            "secondary": "backend_runtime_s from std::chrono",
        },
        "fresh_process_per_replica": True,
        "outputs": {key: str(value) for key, value in paths.items()},
        "status": "initialized",
        "preflight": [],
        "skipped_block_sizes": [],
    }


def output_paths(tag):
    return {
        "raw_csv": RESULTS_DIR / f"cuda_tiled_block_size_raw_{tag}.csv",
        "summary_csv": RESULTS_DIR / f"cuda_tiled_block_size_summary_{tag}.csv",
        "metadata_json": RESULTS_DIR / f"cuda_tiled_block_size_metadata_{tag}.json",
    }


def prepare_outputs(paths, overwrite):
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing output(s):\n  {joined}\nUse --overwrite.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_command(n, steps, block_size):
    return [
        str(BIN), "--benchmark", "--N", str(n), "--steps", str(steps),
        "--dt", DT_ARGUMENT, "--epsilon", str(EPSILON), "--init", "random",
        "--seed", str(SEED), "--integrator", "verlet",
        "--block-size", str(block_size),
    ]


def parse_benchmark_output(stdout, expected_steps):
    values = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition(",")
        if separator and key in {
            "compute_runtime_s", "backend_runtime_s", "benchmark_steps"
        }:
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


def run_once(n, block_size, steps, timeout, phase):
    command = benchmark_command(n, steps, block_size)
    print(
        f"[{phase}] N={n} block_size={block_size}: {shlex.join(command)}",
        flush=True,
    )
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=timeout
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


def coefficient_of_variation(mean, std):
    return std / mean if mean else 0.0


def summarize(raw_rows, n_values, block_sizes):
    rows = []
    by_key = {}
    for n in n_values:
        for block_size in block_sizes:
            selected = [
                row for row in raw_rows
                if row["N"] == n and row["block_size"] == block_size
            ]
            compute = [row["compute_runtime_s"] for row in selected]
            backend = [row["backend_runtime_s"] for row in selected]
            throughput = [row["effective_interactions_per_s"] for row in selected]
            mean_compute = statistics.mean(compute)
            std_compute = sample_std(compute)
            mean_backend = statistics.mean(backend)
            std_backend = sample_std(backend)
            mean_throughput = statistics.mean(throughput)
            std_throughput = sample_std(throughput)
            row = {
                "backend": "cuda_tiled", "N": n, "block_size": block_size,
                "steps": selected[0]["steps"],
                "force_evaluations": selected[0]["force_evaluations"],
                "pair_iterations": selected[0]["pair_iterations"],
                "physical_nonself_pairs": selected[0]["physical_nonself_pairs"],
                "repeats": len(selected),
                "mean_compute_runtime_s": mean_compute,
                "std_compute_runtime_s": std_compute,
                "median_compute_runtime_s": statistics.median(compute),
                "cv_compute_runtime": coefficient_of_variation(mean_compute, std_compute),
                "mean_backend_runtime_s": mean_backend,
                "std_backend_runtime_s": std_backend,
                "median_backend_runtime_s": statistics.median(backend),
                "cv_backend_runtime": coefficient_of_variation(mean_backend, std_backend),
                "mean_effective_interactions_per_s": mean_throughput,
                "std_effective_interactions_per_s": std_throughput,
                "median_effective_interactions_per_s": statistics.median(throughput),
                "cv_effective_interactions_per_s": coefficient_of_variation(
                    mean_throughput, std_throughput
                ),
            }
            rows.append(row)
            by_key[(n, block_size)] = row

    for n in n_values:
        minimum = min(by_key[(n, block)]["mean_compute_runtime_s"] for block in block_sizes)
        for block in block_sizes:
            by_key[(n, block)]["normalized_compute_runtime"] = (
                by_key[(n, block)]["mean_compute_runtime_s"] / minimum
            )

    ranking = []
    for block in block_sizes:
        normalized = [by_key[(n, block)]["normalized_compute_runtime"] for n in n_values]
        cvs = [by_key[(n, block)]["cv_compute_runtime"] for n in n_values]
        ranking.append({
            "block_size": block,
            "score": math.exp(statistics.mean(math.log(value) for value in normalized)),
            "mean_cv": statistics.mean(cvs),
        })
    ranking.sort(key=lambda item: item["score"])
    best = ranking[0]
    if len(ranking) == 1:
        selection_status = "single_candidate"
        selection_reason = "Only one supported block size was measured."
    else:
        second = ranking[1]
        relative_gap = second["score"] / best["score"] - 1.0
        variability = math.sqrt(best["mean_cv"] ** 2 + second["mean_cv"] ** 2)
        if relative_gap <= variability:
            selection_status = "no_clear_winner"
            selection_reason = (
                f"Best/second relative gap {relative_gap:.6g} is not larger than "
                f"combined mean CV {variability:.6g}."
            )
        else:
            selection_status = "clear_winner"
            selection_reason = (
                f"Block size {best['block_size']} has geometric normalized score "
                f"{best['score']:.6g}; relative gap to second is {relative_gap:.6g}."
            )

    rank_by_block = {item["block_size"]: (rank, item) for rank, item in enumerate(ranking, 1)}
    for row in rows:
        rank, item = rank_by_block[row["block_size"]]
        row["ranking_position"] = rank
        row["geometric_mean_normalized_runtime"] = item["score"]
        row["mean_compute_cv_across_n"] = item["mean_cv"]
        row["selection_status"] = selection_status
        row["selection_reason"] = selection_reason
    return rows, ranking, selection_status, selection_reason


def main():
    args = parse_args()
    os.chdir(ROOT)
    if not BIN.exists():
        raise SystemExit("Missing bin/nbody_cuda_tiled; run `make cuda-tiled CUDA_ARCH=sm_61`.")
    paths = output_paths(args.output_tag)
    prepare_outputs(paths, args.overwrite)
    metadata = collect_metadata(args, paths)
    write_json(paths["metadata_json"], metadata)

    active_blocks = []
    raw_rows = []
    try:
        for block_size in args.block_sizes:
            try:
                (_, _), command = run_once(
                    513, block_size, 1, args.timeout, "preflight"
                )
                metadata["preflight"].append({
                    "block_size": block_size, "status": "passed",
                    "command": shlex.join(command),
                })
                active_blocks.append(block_size)
            except RuntimeError as error:
                metadata["preflight"].append({
                    "block_size": block_size, "status": "failed", "error": str(error),
                })
                if block_size == 512:
                    metadata["skipped_block_sizes"].append(512)
                    print(f"Skipping unsupported block size 512: {error}", file=sys.stderr)
                else:
                    raise

        for n_index, n in enumerate(args.n_values):
            for block_size in balanced_order(active_blocks, n_index):
                run_once(n, block_size, args.steps, args.timeout, "warm-up")

        sequence = 0
        for replica in range(args.repeats):
            for n_index, n in enumerate(args.n_values):
                order = balanced_order(active_blocks, replica + n_index)
                for block_size in order:
                    (compute, backend), command = run_once(
                        n, block_size, args.steps, args.timeout,
                        f"replica {replica + 1}/{args.repeats}",
                    )
                    sequence += 1
                    force_evaluations = args.steps + 1
                    pair_iterations = force_evaluations * n * n
                    raw_rows.append({
                        "sequence": sequence, "replica": replica + 1,
                        "backend": "cuda_tiled", "N": n,
                        "block_size": block_size, "steps": args.steps,
                        "force_evaluations": force_evaluations,
                        "pair_iterations": pair_iterations,
                        "physical_nonself_pairs": force_evaluations * n * (n - 1),
                        "compute_runtime_s": compute,
                        "backend_runtime_s": backend,
                        "effective_interactions_per_s": pair_iterations / compute,
                        "command": shlex.join(command),
                    })
                    write_csv(paths["raw_csv"], RAW_FIELDS, raw_rows)

        summary, ranking, selection_status, selection_reason = summarize(
            raw_rows, args.n_values, active_blocks
        )
        write_csv(paths["summary_csv"], SUMMARY_FIELDS, summary)
        metadata.update({
            "status": "completed", "block_sizes_measured": active_blocks,
            "ranking": ranking, "selection_status": selection_status,
            "selection_reason": selection_reason,
            "measured_commands": [row["command"] for row in raw_rows],
        })
        write_json(paths["metadata_json"], metadata)

        print("block_size,rank,geometric_normalized_runtime,mean_cv")
        for rank, item in enumerate(ranking, 1):
            print(f"{item['block_size']},{rank},{item['score']:.9g},{item['mean_cv']:.9g}")
        print(f"selection_status,{selection_status}")
        print(f"selection_reason,{selection_reason}")
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
