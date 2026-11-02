#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nbody_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "performance"
PROFILING = RESULTS / "profiling"
CAMPAIGNS = {
    "crossover": (
        RESULTS / "final_tuned_crossover_v2_final_stable_raw.csv",
        RESULTS / "final_tuned_crossover_v2_final_stable_summary.csv",
        RESULTS / "final_tuned_crossover_v2_final_stable_metadata.json",
    ),
    "openmp": (
        RESULTS / "openmp_benchmark_raw_openmp_final.csv",
        RESULTS / "openmp_benchmark_summary_openmp_final.csv",
        RESULTS / "openmp_benchmark_metadata_openmp_final.json",
    ),
    "naive_tuning": (
        RESULTS / "cuda_naive_block_size_v1_final_raw.csv",
        RESULTS / "cuda_naive_block_size_v1_final_summary.csv",
        RESULTS / "cuda_naive_block_size_v1_final_metadata.json",
    ),
    "tiled_tuning": (
        RESULTS / "cuda_tiled_block_size_raw_block_size_final_20260901.csv",
        RESULTS / "cuda_tiled_block_size_summary_block_size_final_20260901.csv",
        RESULTS / "cuda_tiled_block_size_metadata_block_size_final_20260901.json",
    ),
}
TRACES = {
    "CUDA naive B=64": PROFILING / "cuda_naive_b64_N8192_trace.txt",
    "CUDA tiled B=128": PROFILING / "cuda_tiled_b128_N8192_final_trace.txt",
}
PTXAS = {
    "cuda_naive": PROFILING / "cuda_naive_final_ptxas_sm61.txt",
    "cuda_tiled": PROFILING / "cuda_tiled_final_ptxas_sm61.txt",
}
FINAL_BACKENDS = ("serial", "openmp_8", "cuda_naive_b64", "cuda_tiled_b128")
FINAL_N = (512, 1024, 2048, 4096, 8192, 16384, 32768)
BLOCK_SIZES = (32, 64, 128, 256, 512)
OPENMP_N = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
OPENMP_THREADS = (1, 2, 4, 8)
BACKEND_LABELS = {
    "serial": "Serial",
    "openmp_8": "OpenMP (8 threads)",
    "cuda_naive_b64": "CUDA naive (B = 64)",
    "cuda_tiled_b128": "CUDA tiled (B = 128)",
}
BACKEND_COLORS = {
    "serial": "tab:blue",
    "openmp_8": "tab:orange",
    "cuda_naive_b64": "tab:green",
    "cuda_tiled_b128": "tab:red",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and plot final tuned NBody performance campaigns."
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    parser.add_argument(
        "--runtime-variants-only",
        action="store_true",
        help="generate only the four absolute-runtime comparison figures",
    )
    parser.add_argument(
        "--speedup-variants-only",
        action="store_true",
        help="generate only the canonical and two-panel paired-speedup figures",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="generate only the canonical pair-loop throughput figure",
    )
    parser.add_argument(
        "--openmp-scaling-only",
        action="store_true",
        help="generate only the canonical OpenMP strong-scaling figure",
    )
    parser.add_argument(
        "--cuda-tuning-only",
        action="store_true",
        help="generate only the combined CUDA tuning figure and heatmaps",
    )
    parser.add_argument(
        "--remaining-performance-only",
        action="store_true",
        help="generate only throughput, OpenMP scaling, CUDA comparison, and profiling",
    )
    return parser.parse_args()


def load_campaign(paths):
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("missing required input(s): " + ", ".join(map(str, missing)))
    raw_path, summary_path, metadata_path = paths
    with raw_path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    with summary_path.open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if not raw or not summary or metadata.get("status") != "completed":
        raise RuntimeError(f"empty or incomplete campaign: {metadata_path}")
    return {"raw": raw, "summary": summary, "metadata": metadata, "paths": paths}


def number(value, label, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise RuntimeError(f"invalid {label}: {value!r}")
    return result


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-15):
        raise RuntimeError(f"{label}: {actual} != {expected}")


def group_key(name, row):
    if name == "crossover":
        return int(row["N"]), row["backend"]
    if name == "openmp":
        return int(row["N"]), row["backend"], int(row["threads"])
    return int(row["N"]), int(row["block_size"])


def command_options(tokens):
    options = {}
    index = 1
    while index < len(tokens):
        option = tokens[index]
        if option == "--benchmark":
            options[option] = True
            index += 1
            continue
        if index + 1 >= len(tokens):
            raise RuntimeError(f"missing value in benchmark command: {' '.join(tokens)}")
        options[option] = tokens[index + 1]
        index += 2
    return options


def validate_campaign(name, data):
    raw, summary, metadata = data["raw"], data["summary"], data["metadata"]
    raw_path, summary_path, metadata_path = data["paths"]
    groups, seen = {}, set()
    for row in raw:
        n, steps = int(row["N"]), int(row["steps"])
        compute = number(row["compute_runtime_s"], "compute runtime", True)
        backend = number(row["backend_runtime_s"], "backend runtime", True)
        if backend < compute or int(row["force_evaluations"]) != steps + 1:
            raise RuntimeError(f"invalid runtime or force count in {raw_path}")
        if "pair_iterations" in row and int(row["pair_iterations"]) != (steps + 1) * n * n:
            raise RuntimeError(f"invalid pair-loop count in {raw_path}")
        for token in ("--benchmark", "--dt", "--epsilon", "--integrator"):
            if token not in row["command"].split():
                raise RuntimeError(f"missing {token} in command from {raw_path}")
        key = group_key(name, row)
        unique = key + (int(row["replica"]),)
        if unique in seen:
            raise RuntimeError(f"duplicate measurement in {raw_path}: {unique}")
        seen.add(unique); groups.setdefault(key, []).append(row)
    if any(len(rows) != 7 for rows in groups.values()):
        raise RuntimeError(f"campaign does not contain seven repeats: {raw_path}")

    if name == "crossover":
        expected = {(n, backend) for n in FINAL_N for backend in FINAL_BACKENDS}
        if set(groups) != expected or metadata.get("N") != list(FINAL_N):
            raise RuntimeError("final tuned crossover configuration is incomplete")
        if metadata.get("backends") != list(FINAL_BACKENDS):
            raise RuntimeError("final tuned crossover backend set is inconsistent")
        required = {"steps": 20, "repeats": 7, "dt": 5e-5, "epsilon": .01,
                    "seed": 12345, "integrator": "verlet", "initializer": "random"}
        for field, expected_value in required.items():
            if metadata.get(field) != expected_value:
                raise RuntimeError(f"{metadata_path}: unexpected {field}")
        metadata_commands = set(metadata.get("commands", []))
        executable_names = {
            "serial": "nbody_serial",
            "openmp_8": "nbody_openmp",
            "cuda_naive_b64": "nbody_cuda_naive",
            "cuda_tiled_b128": "nbody_cuda_tiled",
        }
        expected_threads = {"serial": 1, "openmp_8": 8,
                            "cuda_naive_b64": 0, "cuda_tiled_b128": 0}
        expected_blocks = {"serial": 0, "openmp_8": 0,
                           "cuda_naive_b64": 64, "cuda_tiled_b128": 128}
        for row in raw:
            command = row["command"]
            tokens = command.split()
            backend = row["backend"]
            options = command_options(tokens)
            if (Path(tokens[0]).name != executable_names[backend] or
                    int(row["threads"]) != expected_threads[backend] or
                    int(row["block_size"]) != expected_blocks[backend] or
                    int(row["exit_code"]) != 0 or command not in metadata_commands or
                    options.get("--benchmark") is not True or
                    options.get("--N") != row["N"] or
                    options.get("--steps") != "20" or
                    options.get("--dt") != "5e-5" or
                    options.get("--epsilon") != "0.01" or
                    options.get("--init") != "random" or
                    options.get("--seed") != "12345" or
                    options.get("--integrator") != "verlet" or
                    (expected_blocks[backend] and
                     options.get("--block-size") != str(expected_blocks[backend]))):
                raise RuntimeError(f"raw/metadata command mismatch: {command}")
    elif name == "openmp":
        expected = {
            (n, backend, threads)
            for n in metadata["N"]
            for backend in ("openmp", "openmp_simd")
            for threads in metadata["threads"]
        }
        expected.update((n, "serial", 1) for n in metadata["N"])
        if set(groups) != expected:
            raise RuntimeError("OpenMP scaling campaign configuration is incomplete")
    else:
        n_values = metadata.get("N", metadata.get("n_values"))
        block_sizes = metadata.get(
            "block_sizes_requested", metadata.get("block_sizes")
        )
        expected = {(n, block) for n in n_values for block in block_sizes}
        if set(groups) != expected or set(block_sizes) != set(BLOCK_SIZES):
            raise RuntimeError(f"CUDA tuning campaign configuration is incomplete: {name}")

    summary_by_key = {group_key(name, row): row for row in summary}
    if len(summary_by_key) != len(summary) or set(summary_by_key) != set(groups):
        raise RuntimeError(f"raw/summary configuration mismatch: {summary_path}")
    for key, rows in groups.items():
        values = [float(row["compute_runtime_s"]) for row in rows]
        aggregate = summary_by_key[key]
        aggregates = [
                ("mean_compute_runtime_s", statistics.mean(values)),
                ("median_compute_runtime_s", statistics.median(values)),
                ("std_compute_runtime_s", statistics.stdev(values)),
        ]
        if "cv_compute_runtime_percent" in aggregate:
            aggregates.append(("cv_compute_runtime_percent",
                               100.0 * statistics.stdev(values) /
                               statistics.mean(values)))
        elif "cv_compute_runtime_s" in aggregate:
            aggregates.append(("cv_compute_runtime_s",
                               statistics.stdev(values) / statistics.mean(values)))
        for field, expected_value in aggregates:
            close(number(aggregate[field], field), expected_value,
                  f"{summary_path} {key} {field}")
    return data


def grouped(rows, *fields):
    output = {}
    for row in rows:
        output.setdefault(tuple(row[field] for field in fields), []).append(row)
    return output


def numeric_n_axis(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([str(value) for value in values], rotation=25, ha="right")


def save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def paired_ratios(raw, numerator, denominator):
    index = {(row["N"], row["backend"], row["replica"]):
             float(row["compute_runtime_s"]) for row in raw}
    return {n: [index[(str(n), numerator, str(replica))] /
                index[(str(n), denominator, str(replica))]
                for replica in range(1, 8)] for n in FINAL_N}


def paired_speedup_statistics(raw):
    result = {}
    for backend in FINAL_BACKENDS[1:]:
        ratios = paired_ratios(raw, "serial", backend)
        for n in FINAL_N:
            q1, _, q3 = statistics.quantiles(
                ratios[n], n=4, method="inclusive"
            )
            median = statistics.median(ratios[n])
            result[(n, backend)] = {
                "median": median,
                "q1": q1,
                "q3": q3,
            }
    return result


def draw_speedup(ax, stats, backend):
    medians = [stats[(n, backend)]["median"] for n in FINAL_N]
    yerr = [
        [medians[i] - stats[(n, backend)]["q1"]
         for i, n in enumerate(FINAL_N)],
        [stats[(n, backend)]["q3"] - medians[i]
         for i, n in enumerate(FINAL_N)],
    ]
    ax.errorbar(
        FINAL_N, medians, yerr=yerr, marker="o", markersize=5,
        linewidth=1.8, elinewidth=.8, capsize=2.5,
        color=BACKEND_COLORS[backend], ecolor=BACKEND_COLORS[backend],
        label=BACKEND_LABELS[backend],
    )
    return medians


def runtime_statistics(raw):
    groups = grouped(raw, "N", "backend")
    result = {}
    for n in FINAL_N:
        for backend in FINAL_BACKENDS:
            values = [float(row["compute_runtime_s"])
                      for row in groups[(str(n), backend)]]
            if len(values) != 7:
                raise RuntimeError(f"expected seven runtimes for {backend}, N={n}")
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
            mean = statistics.mean(values)
            result[(n, backend)] = {
                "mean": mean,
                "median": statistics.median(values),
                "sd": statistics.stdev(values),
                "q1": q1,
                "q3": q3,
                "cv_percent": 100.0 * statistics.stdev(values) / mean,
            }
    return result


def runtime_series(stats, backend, mode, n_values=FINAL_N):
    centers, lower, upper = [], [], []
    for n in n_values:
        item = stats[(n, backend)]
        center = item[mode]
        centers.append(center)
        if mode == "median":
            lower.append(center - item["q1"])
            upper.append(item["q3"] - center)
        else:
            lower.append(item["sd"])
            upper.append(item["sd"])
    return centers, [lower, upper]


def plot_runtime_variants(raw, output_dir, mode):
    stats = runtime_statistics(raw)
    statistic_note = ("Points: median; error bars: IQR" if mode == "median" else
                      "Points: mean; error bars: sample SD")
    filename = ("runtime_vs_N_median_iqr.png" if mode == "median" else
                "runtime_vs_N_mean_sd.png")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for ax, backends, title in (
            (axes[0], FINAL_BACKENDS, "All backends"),
            (axes[1], FINAL_BACKENDS[1:], "Parallel backends")):
        for backend in backends:
            centers, errors = runtime_series(stats, backend, mode)
            ax.errorbar(
                FINAL_N, centers, yerr=errors, marker="o", capsize=3,
                color=BACKEND_COLORS[backend], label=BACKEND_LABELS[backend],
            )
        numeric_n_axis(ax, FINAL_N)
        ax.set_yscale("log")
        ax.set_xlabel("Particles N")
        ax.set_ylabel("Compute runtime [s]")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=.25)
        ax.legend(fontsize=8)

    anchor = stats[(FINAL_N[-1], "serial")][mode]
    axes[0].plot(
        FINAL_N, [anchor * (n / FINAL_N[-1]) ** 2 for n in FINAL_N], ":",
        color="gray", label="N² reference, anchored at largest serial point",
    )
    axes[0].legend(fontsize=8)
    fig.suptitle("Compute runtime versus particle count", fontsize=14)
    fig.text(.5, .925, "Velocity Verlet, 20 steps, 7 repetitions",
             ha="center", fontsize=10)
    fig.text(.5, .012, statistic_note, ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, .04, 1, .90))
    return save(fig, output_dir / filename, tight=False), stats


def plot_large_n_runtime(raw, output_dir, mode):
    stats = runtime_statistics(raw)
    n_values = FINAL_N[-3:]
    backends = FINAL_BACKENDS[1:]
    filename = ("runtime_large_N_median_iqr.png" if mode == "median" else
                "runtime_large_N_mean_sd.png")
    subtitle = ("Velocity Verlet, 20 steps · median and IQR over 7 repetitions"
                if mode == "median" else
                "Velocity Verlet, 20 steps · mean and sample SD over 7 repetitions")
    positions = list(range(len(n_values)))
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    annotation_offsets = {
        "openmp_8": (-62, 10),
        "cuda_naive_b64": (-62, -20),
        "cuda_tiled_b128": (8, 9),
    }
    for backend in backends:
        centers, errors = runtime_series(stats, backend, mode, n_values)
        ax.errorbar(
            positions, centers, yerr=errors, marker="o", capsize=4,
            color=BACKEND_COLORS[backend], label=BACKEND_LABELS[backend],
        )
        ax.annotate(
            f"{centers[-1]:.1f} s", (positions[-1], centers[-1]),
            xytext=annotation_offsets[backend], textcoords="offset points",
            color=BACKEND_COLORS[backend], fontsize=9,
        )
    ax.set_xticks(positions, [str(n) for n in n_values])
    ax.set_xlabel("Particles N")
    ax.set_ylabel("Compute runtime [s]")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Compute runtime at large particle counts", fontsize=14)
    fig.text(.5, .91, subtitle, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .88))
    return save(fig, output_dir / filename, tight=False)


def plot_runtime_comparisons(raw, output_dir):
    median_path, stats = plot_runtime_variants(raw, output_dir, "median")
    mean_path, _ = plot_runtime_variants(raw, output_dir, "mean")
    return [
        median_path,
        mean_path,
        plot_large_n_runtime(raw, output_dir, "median"),
        plot_large_n_runtime(raw, output_dir, "mean"),
    ], stats


def plot_runtime(raw, output_dir):
    groups = grouped(raw, "N", "backend")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for backend in FINAL_BACKENDS:
        means, errors = [], []
        for n in FINAL_N:
            values = [float(row["compute_runtime_s"]) for row in groups[(str(n), backend)]]
            means.append(statistics.mean(values)); errors.append(statistics.stdev(values))
        ax.errorbar(FINAL_N, means, yerr=errors, marker="o", capsize=3,
                    color=BACKEND_COLORS[backend], label=BACKEND_LABELS[backend])
    reference = statistics.median(float(row["compute_runtime_s"])
                                  for row in groups[("4096", "serial")])
    ax.plot(FINAL_N, [reference * (n / 4096) ** 2 for n in FINAL_N], ":",
            color="gray", label=r"$O(N^2)$ reference slope (not a fit)")
    numeric_n_axis(ax, FINAL_N); ax.set_yscale("log")
    ax.set_xlabel("Particles N"); ax.set_ylabel("Compute runtime (s)")
    ax.set_title("Final tuned crossover: Velocity Verlet, 20 steps")
    ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=8)
    return save(fig, output_dir / "final_tuned_runtime_vs_N.png")


def plot_speedup(raw, output_dir):
    stats = paired_speedup_statistics(raw)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.axhline(1, color="gray", linestyle="--", linewidth=1.2,
               label="Serial parity")
    medians = {backend: draw_speedup(ax, stats, backend)
               for backend in FINAL_BACKENDS[1:]}
    annotation_offsets = {
        "openmp_8": (-8, -12),
        "cuda_naive_b64": (-8, 12),
        "cuda_tiled_b128": (-8, -18),
    }
    for backend in FINAL_BACKENDS[1:]:
        ax.annotate(
            f"{medians[backend][-1]:.2f}×",
            (FINAL_N[-1], medians[backend][-1]),
            xytext=annotation_offsets[backend], textcoords="offset points",
            color=BACKEND_COLORS[backend], fontsize=9, ha="right",
        )
    numeric_n_axis(ax, FINAL_N)
    ax.set_xlabel("Particles N")
    ax.set_ylabel(r"Speedup  $T_{serial},/\,T_{backend}$")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Speedup relative to the serial implementation", fontsize=14)
    fig.text(.5, .91,
             "Velocity Verlet, 20 steps · 7 replica-wise paired measurements",
             ha="center", fontsize=10)
    fig.text(.5, .02, "Points: median paired speedup; error bars: IQR",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, .05, 1, .88))
    return save(fig, output_dir / "final_tuned_speedup_vs_serial.png", tight=False)


def plot_speedup_two_panel(raw, output_dir):
    stats = paired_speedup_statistics(raw)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    panels = (
        (axes[0], ("openmp_8", "cuda_naive_b64"),
         "OpenMP versus CUDA naive"),
        (axes[1], ("openmp_8", "cuda_tiled_b128"),
         "OpenMP versus CUDA tiled"),
    )
    for ax, backends, title in panels:
        ax.axhline(1, color="gray", linestyle="--", linewidth=1.2,
                   label="Serial parity")
        for backend in backends:
            draw_speedup(ax, stats, backend)
        numeric_n_axis(ax, FINAL_N)
        ax.set_xlabel("Particles N")
        ax.set_title(title)
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel(r"Speedup  $T_{serial},/\,T_{backend}$")
    fig.suptitle("Speedup relative to the serial implementation", fontsize=14)
    fig.text(.5, .91,
             "Velocity Verlet, 20 steps · median and IQR of 7 paired measurements",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .88))
    return save(
        fig, output_dir / "speedup_vs_serial_two_panel_median_iqr.png", tight=False
    )


def throughput_statistics(raw):
    groups = grouped(raw, "N", "backend")
    result = {}
    for n in FINAL_N:
        for backend in FINAL_BACKENDS:
            rows = groups[(str(n), backend)]
            values = [int(row["pair_iterations"]) /
                      float(row["compute_runtime_s"]) for row in rows]
            if len(values) != 7:
                raise RuntimeError(f"expected seven throughputs for {backend}, N={n}")
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
            mean = statistics.mean(values)
            result[(n, backend)] = {
                "mean": mean,
                "median": statistics.median(values),
                "sd": statistics.stdev(values),
                "q1": q1,
                "q3": q3,
                "cv_percent": 100.0 * statistics.stdev(values) / mean,
            }
    return result


def plot_throughput(raw, output_dir):
    stats = throughput_statistics(raw)
    scale = 1e8
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for backend in FINAL_BACKENDS:
        medians = [stats[(n, backend)]["median"] / scale for n in FINAL_N]
        lower = [(stats[(n, backend)]["median"] -
                  stats[(n, backend)]["q1"]) / scale for n in FINAL_N]
        upper = [(stats[(n, backend)]["q3"] -
                  stats[(n, backend)]["median"]) / scale for n in FINAL_N]
        if any(value < 0.0 for value in lower + upper):
            raise RuntimeError(f"negative throughput error bar for {backend}")
        ax.errorbar(
            FINAL_N, medians, yerr=[lower, upper], marker="o", markersize=5,
            linewidth=1.8, elinewidth=.8, capsize=2.5,
            color=BACKEND_COLORS[backend], ecolor=BACKEND_COLORS[backend],
            label=BACKEND_LABELS[backend],
        )
    numeric_n_axis(ax, FINAL_N)
    ax.set_xlabel("Particles N")
    ax.set_ylabel("Pair-loop visits per second [×10⁸]")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("Effective pair-loop throughput", fontsize=14)
    fig.text(.5, .91, "Velocity Verlet, 20 steps · 21N² loop visits per run",
             ha="center", fontsize=10)
    fig.text(.5, .02, "Points: median; error bars: IQR",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, .05, 1, .88))
    return (save(fig, output_dir / "final_tuned_pair_loop_throughput.png",
                 tight=False), stats)


def openmp_scaling_statistics(raw):
    rows = [row for row in raw if row["backend"] == "openmp"]
    index = {(int(row["N"]), int(row["threads"]), int(row["replica"])):
             float(row["compute_runtime_s"]) for row in rows}
    expected = {(n, threads, replica) for n in OPENMP_N
                for threads in OPENMP_THREADS for replica in range(1, 8)}
    if set(index) != expected:
        raise RuntimeError("incomplete OpenMP replica pairing")
    result = {}
    for n in OPENMP_N:
        for thread_count in OPENMP_THREADS[1:]:
            ratios = [index[(n, 1, replica)] / index[(n, thread_count, replica)]
                      for replica in range(1, 8)]
            q1, _, q3 = statistics.quantiles(ratios, n=4, method="inclusive")
            result[(n, thread_count)] = {
                "median": statistics.median(ratios),
                "q1": q1,
                "q3": q3,
                "efficiency": statistics.median(
                    ratio / thread_count for ratio in ratios
                ),
            }
    return result


def plot_openmp(raw, output_dir):
    stats = openmp_scaling_statistics(raw)
    thread_colors = {2: "tab:orange", 4: "tab:green", 8: "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].axhline(1, color="gray", linestyle="--", linewidth=1.2,
                    label="One-thread baseline")
    matrix = []
    for thread_count in OPENMP_THREADS[1:]:
        medians = [stats[(n, thread_count)]["median"] for n in OPENMP_N]
        lower = [stats[(n, thread_count)]["median"] -
                 stats[(n, thread_count)]["q1"] for n in OPENMP_N]
        upper = [stats[(n, thread_count)]["q3"] -
                 stats[(n, thread_count)]["median"] for n in OPENMP_N]
        color = thread_colors[thread_count]
        axes[0].errorbar(
            OPENMP_N, medians, yerr=[lower, upper], marker="o", markersize=5,
            linewidth=1.8, elinewidth=.8, capsize=2.5, color=color,
            ecolor=to_rgba(color, .55), label=f"{thread_count} threads",
        )
        matrix.append([stats[(n, thread_count)]["efficiency"] for n in OPENMP_N])
    numeric_n_axis(axes[0], OPENMP_N)
    axes[0].set_xlabel("Particles N")
    axes[0].set_ylabel(r"Speedup  $T_1\,/\,T_p$")
    axes[0].grid(True, alpha=.25)
    axes[0].legend(fontsize=8, loc="upper left")
    image = axes[1].imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[1].set_xticks(range(len(OPENMP_N)), [str(n) for n in OPENMP_N], rotation=30)
    axes[1].set_yticks(range(3), ["2", "4", "8"])
    axes[1].set_xlabel("Particles N"); axes[1].set_ylabel("OpenMP threads")
    for i, values in enumerate(matrix):
        for j, value in enumerate(values):
            axes[1].text(j, i, f"{value:.2f}", ha="center", va="center",
                         color="white" if value < .55 else "black", fontsize=7)
    fig.colorbar(image, ax=axes[1]).set_label(r"Efficiency $S_p / p$")
    axes[0].set_title("Paired speedup relative to OpenMP-1")
    axes[1].set_title(r"Parallel efficiency  $E_p = S_p / p$")
    fig.suptitle("OpenMP strong scaling", fontsize=14)
    fig.text(.5, .915,
             "Paired measurements relative to the one-thread OpenMP baseline "
             "· 7 paired repetitions",
             ha="center", fontsize=9)
    fig.text(.25, .01, "Points: median paired speedup; error bars: IQR",
             ha="center", fontsize=8, color="dimgray")
    fig.text(.74, .01,
             "Values slightly above 1 are not interpreted as demonstrated "
             "superlinear scaling.",
             ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, .045, 1, .88))
    return save(fig, output_dir / "openmp_scaling_internal_baseline.png",
                tight=False)


def plot_tuning(raw, title, filename, output_dir):
    groups = grouped(raw, "N", "block_size")
    n_values = sorted({int(row["N"]) for row in raw})
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for n in n_values:
        means, stds = [], []
        for block in BLOCK_SIZES:
            values = [float(row["compute_runtime_s"])
                      for row in groups[(str(n), str(block))]]
            means.append(statistics.mean(values)); stds.append(statistics.stdev(values))
        fastest = min(means)
        ax.errorbar(BLOCK_SIZES, [value / fastest for value in means],
                    yerr=[value / fastest for value in stds], marker="o", capsize=3,
                    label=f"N={n}")
    ax.axhline(1, color="gray", linestyle="--")
    ax.set_xticks(BLOCK_SIZES); ax.set_xlabel("CUDA block size (threads)")
    ax.set_ylabel("Mean runtime / fastest mean runtime at the same N")
    ax.set_title(title + "\nError bars: sample SD / fastest mean at each N")
    ax.grid(True, alpha=.25); ax.legend(fontsize=8)
    return save(fig, output_dir / filename)


def tuning_statistics(raw):
    groups = grouped(raw, "N", "block_size")
    n_values = sorted({int(row["N"]) for row in raw})
    result = {}
    for n in n_values:
        absolute = {}
        for block in BLOCK_SIZES:
            values = [float(row["compute_runtime_s"])
                      for row in groups[(str(n), str(block))]]
            if len(values) != 7:
                raise RuntimeError(f"expected seven tuning runs for N={n}, B={block}")
            absolute[block] = (statistics.mean(values), statistics.stdev(values))
        fastest_mean = min(mean for mean, _ in absolute.values())
        for block, (mean, sample_sd) in absolute.items():
            result[(n, block)] = {
                "relative_runtime": mean / fastest_mean,
                "normalized_sample_sd": sample_sd / fastest_mean,
            }
        if min(result[(n, block)]["relative_runtime"]
               for block in BLOCK_SIZES) != 1.0:
            raise RuntimeError(f"CUDA tuning normalization failed for N={n}")
    return n_values, result


def plot_tuning_with_uncertainty(naive_raw, tiled_raw, output_dir):
    naive_n, naive_stats = tuning_statistics(naive_raw)
    tiled_n, tiled_stats = tuning_statistics(tiled_raw)
    all_n = sorted(set(naive_n) | set(tiled_n))
    cmap = plt.get_cmap("tab10")
    colors = {n: cmap(index) for index, n in enumerate(all_n)}
    offsets = {n: (index - (len(all_n) - 1) / 2) * .10
               for index, n in enumerate(all_n)}
    positions = list(range(len(BLOCK_SIZES)))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
    for ax, n_values, stats, title in (
            (axes[0], naive_n, naive_stats, "CUDA naive"),
            (axes[1], tiled_n, tiled_stats, "CUDA tiled")):
        ax.axhline(1, color="gray", linestyle="--", linewidth=1.1)
        for n in n_values:
            x_values = [position + offsets[n] for position in positions]
            centers = [stats[(n, block)]["relative_runtime"]
                       for block in BLOCK_SIZES]
            errors = [stats[(n, block)]["normalized_sample_sd"]
                      for block in BLOCK_SIZES]
            ax.errorbar(
                x_values, centers, yerr=errors, fmt="o", markersize=5,
                elinewidth=.8, capsize=2.5, color=colors[n],
                ecolor=to_rgba(colors[n], .55), label=f"N={n}",
            )
        ax.set_xticks(positions, [str(block) for block in BLOCK_SIZES])
        ax.set_xlabel("Block size B [threads]")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Relative runtime")
    fig.suptitle("CUDA block-size sensitivity", fontsize=14)
    fig.text(.5, .91,
             "Mean runtime normalized within each backend and particle count "
             "· 7 repetitions",
             ha="center", fontsize=9)
    fig.text(.5, .025,
             "1 = fastest mean runtime at the same backend and N    ·    "
             "Error bars: normalized sample SD",
             ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, .055, 1, .88))
    return save(fig, output_dir / "cuda_block_size_tuning_with_uncertainty.png",
                tight=False)


def plot_tuning_heatmaps(naive_raw, tiled_raw, output_dir):
    fig = plt.figure(figsize=(12, 5.2))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, .045),
                            left=.07, right=.95, bottom=.16, top=.82,
                            wspace=.08)
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    axes[1].sharey(axes[0])
    colorbar_ax = fig.add_subplot(grid[0, 2])
    matrices = []
    for raw in (naive_raw, tiled_raw):
        n_values, stats = tuning_statistics(raw)
        matrix = [[stats[(n, block)]["relative_runtime"] for n in n_values]
                  for block in BLOCK_SIZES]
        matrices.append((matrix, n_values))
    vmax = max(value for matrix, _ in matrices for row in matrix for value in row)
    for ax, (matrix, n_values), title in zip(
            axes, matrices, ("CUDA naive", "CUDA tiled")):
        image = ax.imshow(matrix, aspect="auto", vmin=1, vmax=vmax, cmap="viridis_r")
        ax.set_xticks(range(len(n_values)), [str(n) for n in n_values], rotation=30)
        ax.set_yticks(range(len(BLOCK_SIZES)), [str(block) for block in BLOCK_SIZES])
        ax.set_xlabel("Particles N"); ax.set_title(title)
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                ax.text(column_index, row_index, f"{value:.3f}", ha="center",
                        va="center", fontsize=7,
                        color="white" if value > 1 + (vmax - 1) / 2 else "black")
    axes[0].set_ylabel("Block size B [threads]")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.colorbar(image, cax=colorbar_ax, label="Relative runtime")
    fig.suptitle("CUDA block-size sensitivity", fontsize=14, y=.98)
    fig.text(.5, .925,
             "Mean runtime normalized within each backend and particle count "
             "· lower is better",
             ha="center", fontsize=9)
    fig.text(.5, .015, "1 = fastest mean runtime at the same backend and N",
             ha="center", fontsize=8, color="dimgray")
    return save(fig, output_dir / "cuda_block_size_tuning_heatmaps.png", tight=False)


def plot_naive_tiled(raw, output_dir):
    ratios = paired_ratios(raw, "cuda_tiled_b128", "cuda_naive_b64")
    differences = {n: [100.0 * (ratio - 1.0) for ratio in ratios[n]]
                   for n in FINAL_N}
    medians = [statistics.median(differences[n]) for n in FINAL_N]
    quartiles = [statistics.quantiles(differences[n], n=4, method="inclusive")
                 for n in FINAL_N]
    yerr = [[medians[i] - quartiles[i][0] for i in range(len(FINAL_N))],
            [quartiles[i][2] - medians[i] for i in range(len(FINAL_N))]]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.errorbar(FINAL_N, medians, yerr=yerr, marker="o", markersize=5,
                linewidth=1.8, elinewidth=.8, capsize=2.5, color="tab:blue",
                ecolor=to_rgba("tab:blue", .55))
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.1)
    numeric_n_axis(ax, FINAL_N); ax.set_xlabel("Particles N")
    ax.set_ylabel("Tiled runtime difference [%]")
    ax.grid(True, alpha=.25)
    y_min, y_max = ax.get_ylim()
    ax.text(FINAL_N[0], y_max - .08 * (y_max - y_min), "Tiled slower",
            color="dimgray", va="top")
    ax.text(FINAL_N[0], y_min + .08 * (y_max - y_min), "Tiled faster",
            color="dimgray", va="bottom")
    annotation_offsets = {
        512: (0, 8), 1024: (0, -16), 2048: (0, -16), 4096: (0, 8),
        8192: (0, 8), 16384: (0, 8), 32768: (0, 8),
    }
    for n in FINAL_N:
        index = FINAL_N.index(n)
        ax.annotate(f"{medians[index]:.1f}%", (n, medians[index]),
                    xytext=annotation_offsets[n], textcoords="offset points",
                    ha="center", fontsize=8, color="#17365d")
    fig.suptitle("CUDA tiled versus naive", fontsize=14)
    fig.text(.5, .91,
             "Paired runtime difference · tiled B=128 relative to naive B=64 "
             "· 7 repetitions",
             ha="center", fontsize=9)
    fig.text(.5, .02, "Points: median; error bars: IQR",
             ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, .05, 1, .88))
    return (save(fig, output_dir / "final_tuned_cuda_tiled_over_naive.png",
                 tight=False), ratios, differences)


def duration_seconds(text):
    match = re.fullmatch(r"([0-9.]+)(ns|us|ms|s)", text)
    if not match:
        raise RuntimeError(f"unsupported nvprof duration: {text}")
    return float(match.group(1)) * {"ns": 1e-9, "us": 1e-6,
                                   "ms": 1e-3, "s": 1}[match.group(2)]


def trace_breakdown(path):
    values = {"acceleration": [], "position": [], "velocity": [], "HtoD": []}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2 or not re.fullmatch(r"[0-9.]+(?:ns|us|ms|s)", parts[1]):
            continue
        key = ("acceleration" if "compute_acceleration" in line else
               "position" if "update_positions" in line else
               "velocity" if "update_velocities" in line else
               "HtoD" if "memcpy HtoD" in line else None)
        if key:
            values[key].append(duration_seconds(parts[1]))
    expected = {"acceleration": 21, "position": 20, "velocity": 20, "HtoD": 7}
    counts = {key: len(entries) for key, entries in values.items()}
    if counts != expected:
        raise RuntimeError(f"unexpected nvprof operation counts in {path}: {counts}")
    return {key: sum(entries) for key, entries in values.items()}


def final_profiling_data():
    missing = [path for path in (*TRACES.values(), *PTXAS.values())
               if not path.exists()]
    if missing:
        raise RuntimeError("missing final profiling input(s): " +
                           ", ".join(map(str, missing)))
    traces = {name: trace_breakdown(path) for name, path in TRACES.items()}
    registers = {}
    for backend, path in PTXAS.items():
        entries = parse_ptxas(path, backend)
        registers[backend] = next(
            entry["registers_per_thread"] for entry in entries
            if entry["kernel"] == "acceleration"
        )
    occupancy_path = PROFILING / "cuda_occupancy_theoretical.csv"
    if not occupancy_path.exists():
        raise RuntimeError(f"missing final occupancy report: {occupancy_path}")
    with occupancy_path.open(newline="") as handle:
        occupancy = {row["backend"]: row for row in csv.DictReader(handle)}
    if set(occupancy) != {"cuda_naive", "cuda_tiled"}:
        raise RuntimeError("incomplete final occupancy report")
    return traces, registers, occupancy


def plot_final_profiling_summary(output_dir):
    traces, registers, occupancy = final_profiling_data()
    names = ("CUDA naive B=64", "CUDA tiled B=128")
    backends = ("cuda_naive", "cuda_tiled")
    acceleration = {name: traces[name]["acceleration"] for name in names}
    naive_time = acceleration[names[0]]
    total_profiled = {name: sum(traces[name].values()) for name in names}
    values = (
        ("Accumulated acceleration-kernel time [s]",
         *(f"{acceleration[name]:.3f}" for name in names)),
        ("Runtime difference relative to naive [%]", "0.00",
         f"{100.0 * (acceleration[names[1]] / naive_time - 1.0):.2f}"),
        ("Acceleration share of profiled GPU time [%]",
         *(f"{100.0 * acceleration[name] / total_profiled[name]:.3f}"
           for name in names)),
        ("Acceleration-kernel calls", "21", "21"),
        ("Position-kernel calls", "20", "20"),
        ("Velocity-kernel calls", "20", "20"),
        ("Host-to-device copies", "7", "7"),
        ("Registers per acceleration thread",
         *(str(registers[backend]) for backend in backends)),
        ("Static shared memory per block [B]", "0", "0"),
        ("Dynamic shared memory per block [B]", "0", "4096"),
        ("Theoretical occupancy",
         *(f"{100.0 * float(occupancy[backend]['theoretical_occupancy']):.2f}%"
           for backend in backends)),
        ("Limiting resource",
         *(occupancy[backend]["limiting_resource"] for backend in backends)),
    )
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    ax.axis("off")
    table = ax.table(
        cellText=values,
        colLabels=("Metric", "CUDA naive (B=64)", "CUDA tiled (B=128)"),
        colWidths=(.55, .225, .225), cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    table.scale(1.02, 1.42)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9e2f3")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f5f7")
        if column == 0 and row > 0:
            cell.set_text_props(ha="left")
    fig.suptitle("Final CUDA profiling summary", fontsize=15, y=.96)
    fig.text(.5, .91, "N=8192 · 20 Velocity Verlet steps · nvprof and PTXAS data",
             ha="center", fontsize=10)
    return save(fig, output_dir / "final_cuda_profiling_summary.png", tight=False)


def plot_final_nvprof(output_dir):
    missing = [path for path in TRACES.values() if not path.exists()]
    if missing:
        raise RuntimeError("missing final nvprof trace(s): " + ", ".join(map(str, missing)))
    breakdowns = [trace_breakdown(path) for path in TRACES.values()]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bottoms = [0.0] * len(TRACES)
    for key, label in (("acceleration", "Acceleration"), ("position", "Position"),
                       ("velocity", "Velocity"), ("HtoD", "H2D copies")):
        values = [breakdown[key] for breakdown in breakdowns]
        ax.bar(list(TRACES), values, bottom=bottoms, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    for index, value in enumerate(bottoms):
        ax.text(index, value, f"{value:.3f} s",
                ha="center", va="bottom")
    ax.set_ylabel("Accumulated nvprof time (s)")
    ax.set_title("Final nvprof diagnostic, N=8192, 20 steps")
    ax.grid(True, axis="y", alpha=.25); ax.legend(fontsize=8)
    return save(fig, output_dir / "final_nvprof_naive_b64_vs_tiled_b128.png")


def parse_ptxas(path, backend):
    text = path.read_text()
    entries = []
    chunks = text.split("Compiling entry function '")[1:]
    for chunk in chunks:
        symbol = chunk.split("'", 1)[0]
        properties = chunk.split("Function properties", 1)[1]
        registers = int(re.search(r"Used (\d+) registers", properties).group(1))
        stack, stores, loads = map(int, re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
            properties).groups())
        kernel = ("acceleration" if "compute_acceleration" in symbol else
                  "position update" if "update_positions" in symbol else
                  "velocity update" if "update_velocities" in symbol else
                  "Euler step" if "step_euler" in symbol else "smoke")
        entries.append({"backend": backend, "kernel": kernel, "architecture": "sm_61",
                        "registers_per_thread": registers, "static_shared_bytes": 0,
                        "stack_frame_bytes": stack, "spill_store_bytes": stores,
                        "spill_load_bytes": loads, "source": str(path.relative_to(ROOT))})
    return entries


def write_profiling_summaries():
    missing = [path for path in PTXAS.values() if not path.exists()]
    if missing:
        raise RuntimeError("missing final PTXAS report(s): " + ", ".join(map(str, missing)))
    entries = [entry for backend, path in PTXAS.items()
               for entry in parse_ptxas(path, backend)]
    ptxas_path = PROFILING / "cuda_ptxas_summary.csv"
    with ptxas_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader(); writer.writerows(entries)

    acceleration = {entry["backend"]: entry["registers_per_thread"] for entry in entries
                    if entry["kernel"] == "acceleration"}
    occupancy_rows = []
    for backend, block, dynamic_shared in (("cuda_naive", 64, 0),
                                            ("cuda_tiled", 128, 4 * 128 * 8)):
        registers = acceleration[backend]
        warps = math.ceil(block / 32)
        registers_per_warp = math.ceil(registers * 32 / 256) * 256
        limits = {"threads": 2048 // block,
                  "registers": 65536 // (registers_per_warp * warps),
                  "shared_memory": 32 if dynamic_shared == 0 else 49152 // dynamic_shared,
                  "architecture": 32}
        resident = min(limits.values())
        occupancy_rows.append({
            "backend": backend, "compute_capability": "6.1", "block_size": block,
            "registers_per_thread": registers,
            "dynamic_shared_bytes_per_block": dynamic_shared,
            "resident_blocks_per_sm": resident, "active_warps_per_sm": resident * warps,
            "theoretical_occupancy": resident * warps / 64,
            "limiting_resource": "+".join(key for key, value in limits.items()
                                                 if value == resident),
        })
    occupancy_path = PROFILING / "cuda_occupancy_theoretical.csv"
    with occupancy_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(occupancy_rows[0]))
        writer.writeheader(); writer.writerows(occupancy_rows)
    return ptxas_path, occupancy_path


def main():
    args = parse_args()
    plot_only_modes = (args.runtime_variants_only, args.speedup_variants_only,
                       args.throughput_only, args.openmp_scaling_only,
                       args.cuda_tuning_only, args.remaining_performance_only)
    if sum(plot_only_modes) > 1:
        raise RuntimeError("select at most one plot-only mode")
    if args.runtime_variants_only:
        crossover = validate_campaign("crossover", load_campaign(CAMPAIGNS["crossover"]))
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        figures, stats = plot_runtime_comparisons(crossover["raw"], args.figures_dir)
        print("backend,N,mean,median,sample_sd,Q1,Q3,CV_percent")
        for n in FINAL_N:
            for backend in FINAL_BACKENDS:
                values = stats[(n, backend)]
                print(
                    f"{backend},{n},{values['mean']:.17g},{values['median']:.17g},"
                    f"{values['sd']:.17g},{values['q1']:.17g},{values['q3']:.17g},"
                    f"{values['cv_percent']:.17g}"
                )
        for path in figures:
            print(path)
        return
    if args.throughput_only:
        crossover = validate_campaign("crossover", load_campaign(CAMPAIGNS["crossover"]))
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        figure, stats = plot_throughput(crossover["raw"], args.figures_dir)
        print("backend,N,mean,median,sample_sd,Q1,Q3,CV_percent")
        for n in FINAL_N:
            for backend in FINAL_BACKENDS:
                values = stats[(n, backend)]
                print(
                    f"{backend},{n},{values['mean']:.17g},{values['median']:.17g},"
                    f"{values['sd']:.17g},{values['q1']:.17g},{values['q3']:.17g},"
                    f"{values['cv_percent']:.17g}"
                )
        print(figure)
        return
    if args.cuda_tuning_only:
        naive = validate_campaign("naive_tuning", load_campaign(CAMPAIGNS["naive_tuning"]))
        tiled = validate_campaign("tiled_tuning", load_campaign(CAMPAIGNS["tiled_tuning"]))
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        figures = [
            plot_tuning_with_uncertainty(naive["raw"], tiled["raw"],
                                         args.figures_dir),
            plot_tuning_heatmaps(naive["raw"], tiled["raw"], args.figures_dir),
        ]
        for backend, data in (("naive", naive), ("tiled", tiled)):
            n_values, stats = tuning_statistics(data["raw"])
            print("backend,N,block_size,relative_runtime,normalized_sample_sd")
            for n in n_values:
                for block in BLOCK_SIZES:
                    values = stats[(n, block)]
                    print(f"{backend},{n},{block},"
                          f"{values['relative_runtime']:.17g},"
                          f"{values['normalized_sample_sd']:.17g}")
        for figure in figures:
            print(figure)
        return
    if args.openmp_scaling_only:
        campaign = validate_campaign("openmp", load_campaign(CAMPAIGNS["openmp"]))
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        stats = openmp_scaling_statistics(campaign["raw"])
        figure = plot_openmp(campaign["raw"], args.figures_dir)
        print("N,threads,median_paired_speedup,Q1,Q3,median_efficiency")
        for n in OPENMP_N:
            for thread_count in OPENMP_THREADS[1:]:
                values = stats[(n, thread_count)]
                print(f"{n},{thread_count},{values['median']:.17g},"
                      f"{values['q1']:.17g},{values['q3']:.17g},"
                      f"{values['efficiency']:.17g}")
        print(figure)
        return
    if args.remaining_performance_only:
        crossover = validate_campaign("crossover", load_campaign(CAMPAIGNS["crossover"]))
        openmp = validate_campaign("openmp", load_campaign(CAMPAIGNS["openmp"]))
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        throughput, throughput_stats = plot_throughput(
            crossover["raw"], args.figures_dir
        )
        openmp_figure = plot_openmp(openmp["raw"], args.figures_dir)
        comparison, ratios, differences = plot_naive_tiled(
            crossover["raw"], args.figures_dir
        )
        profiling = plot_final_profiling_summary(args.figures_dir)
        print("throughput_N32768_backend,median,Q1,Q3")
        for backend in FINAL_BACKENDS:
            values = throughput_stats[(32768, backend)]
            print(f"{backend},{values['median']:.17g},{values['q1']:.17g},"
                  f"{values['q3']:.17g}")
        print("openmp_N16384_threads,median_speedup,Q1,Q3,median_efficiency")
        openmp_stats = openmp_scaling_statistics(openmp["raw"])
        for thread_count in OPENMP_THREADS[1:]:
            values = openmp_stats[(16384, thread_count)]
            print(f"{thread_count},{values['median']:.17g},{values['q1']:.17g},"
                  f"{values['q3']:.17g},{values['efficiency']:.17g}")
        print("cuda_N1024_replica_ratios," +
              ",".join(f"{value:.17g}" for value in ratios[1024]))
        q1, _, q3 = statistics.quantiles(ratios[1024], n=4, method="inclusive")
        print(f"cuda_N1024_ratio_summary,{statistics.median(ratios[1024]):.17g},"
              f"{q1:.17g},{q3:.17g}")
        for path in (throughput, openmp_figure, comparison, profiling):
            print(path)
        return
    if args.speedup_variants_only:
        crossover = validate_campaign("crossover", load_campaign(CAMPAIGNS["crossover"]))
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        stats = paired_speedup_statistics(crossover["raw"])
        figures = [
            plot_speedup(crossover["raw"], args.figures_dir),
            plot_speedup_two_panel(crossover["raw"], args.figures_dir),
        ]
        print("backend,N,median_paired_speedup,Q1,Q3")
        for n in FINAL_N:
            for backend in FINAL_BACKENDS[1:]:
                values = stats[(n, backend)]
                print(f"{backend},{n},{values['median']:.17g},"
                      f"{values['q1']:.17g},{values['q3']:.17g}")
        for path in figures:
            print(path)
        return
    campaigns = {name: validate_campaign(name, load_campaign(paths))
                 for name, paths in CAMPAIGNS.items()}
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_runtime(campaigns["crossover"]["raw"], args.figures_dir),
        plot_speedup(campaigns["crossover"]["raw"], args.figures_dir),
        plot_throughput(campaigns["crossover"]["raw"], args.figures_dir)[0],
        plot_openmp(campaigns["openmp"]["raw"], args.figures_dir),
        plot_tuning(campaigns["naive_tuning"]["raw"], "CUDA naive block-size tuning",
                    "cuda_naive_block_size_tuning.png", args.figures_dir),
        plot_tuning(campaigns["tiled_tuning"]["raw"], "CUDA tiled block-size tuning",
                    "cuda_tiled_block_size_tuning.png", args.figures_dir),
        plot_tuning_heatmaps(campaigns["naive_tuning"]["raw"],
                             campaigns["tiled_tuning"]["raw"], args.figures_dir),
        plot_naive_tiled(campaigns["crossover"]["raw"], args.figures_dir)[0],
        plot_final_nvprof(args.figures_dir),
    ]
    profiling_summaries = write_profiling_summaries()
    print(f"validated_campaigns,{len(campaigns)}")
    for path in figures:
        print(path)
    for path in profiling_summaries:
        print(path)


if __name__ == "__main__":
    main()
