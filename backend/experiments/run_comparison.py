"""
GA vs Fixed-Time Comparison Experiment
=======================================
Runs both controllers under identical traffic conditions and produces
comparison graphs for the project report.

Outputs (saved to backend/experiments/results/):
    - avg_wait_time_comparison.png
    - queue_length_over_time.png
    - throughput_comparison.png
    - results_summary.txt

Usage:
    cd backend
    python -m experiments.run_comparison
    python -m experiments.run_comparison --hours 2 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from core.intersection import Intersection
from core.ga_controller import GAController, GAConfig
from core.fixed_time_controller import FixedTimeController
from data_pipeline.traffic_generator import TrafficGenerator
from metrics.collector import MetricsCollector
from core.vehicle_physics import YELLOW_DURATION


def run_simulation(
    controller,
    controller_mode: str,
    traffic_gen: TrafficGenerator,
    sim_hours: float,
    tick_duration: float = 1.0,
    ga_evolve_interval: float = 30.0,
    seed: int = 42,
    start_hour: int = 6,
) -> MetricsCollector:
    """
    Run a headless simulation (no Flask, no threads) for the specified
    duration and return the metrics collector.
    """
    intersection = Intersection(
        ns_green_duration=30.0,
        ew_green_duration=30.0,
        tick_duration=tick_duration,
    )

    metrics = MetricsCollector()
    total_ticks = int(sim_hours * 3600 / tick_duration)
    warm_up = int(60 / tick_duration)  # 60 seconds warm-up

    sim_time = 0.0
    last_ga_evolve = 0.0

    for tick in range(total_ticks):
        sim_time += tick_duration
        current_hour = (start_hour + int(sim_time / 3600)) % 24

        # 1. Generate arrivals
        arrivals = traffic_gen.get_arrivals(current_hour, tick_duration)
        intersection.add_vehicles(arrivals)

        # 2. GA evolution (periodic)
        if controller_mode == "ga":
            if sim_time - last_ga_evolve >= ga_evolve_interval:
                last_ga_evolve = sim_time
                ns, ew = controller.evolve(intersection.queues)
                intersection.set_timings(ns, ew)

        # 3. Tick intersection
        state = intersection.tick()

        # 4. Record metrics (skip warm-up)
        if tick >= warm_up:
            metrics.record(state)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="GA vs Fixed-Time Comparison")
    parser.add_argument("--hours", type=float, default=1.0, help="Simulated hours (default 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per controller")
    parser.add_argument("--start-hour", type=int, default=6, help="Start hour of day (default 6)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  GA vs Fixed-Time — Comparison Experiment")
    print("=" * 60)
    print(f"  Duration  : {args.hours} simulated hour(s) per run")
    print(f"  Runs      : {args.runs} per controller")
    print(f"  Base Seed : {args.seed}")
    print("=" * 60)

    # ── Run experiments ─────────────────────────────────────────────
    ga_results = []
    fixed_results = []

    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        traffic_gen = TrafficGenerator(random_seed=seed)

        # GA run
        print(f"\n[Run {run_idx + 1}/{args.runs}] GA Controller (seed={seed})...")
        t0 = time.time()
        ga_controller = GAController(config=GAConfig(random_seed=seed))
        ga_metrics = run_simulation(
            controller=ga_controller,
            controller_mode="ga",
            traffic_gen=TrafficGenerator(random_seed=seed),  # fresh gen with same seed
            sim_hours=args.hours,
            seed=seed,
            start_hour=args.start_hour,
        )
        ga_time = time.time() - t0
        ga_summary = ga_metrics.summary()
        ga_summary["run_time_s"] = round(ga_time, 2)
        ga_results.append(ga_summary)
        print(f"       Avg Wait: {ga_summary['avg_wait_time']}s, "
              f"Throughput: {ga_summary['total_throughput']} veh, "
              f"Compute: {ga_time:.1f}s")

        # Fixed-Time run
        print(f"[Run {run_idx + 1}/{args.runs}] Fixed-Time Controller (seed={seed})...")
        t0 = time.time()
        fixed_controller = FixedTimeController(ns_green=30.0, ew_green=30.0)
        fixed_metrics = run_simulation(
            controller=fixed_controller,
            controller_mode="fixed",
            traffic_gen=TrafficGenerator(random_seed=seed),
            sim_hours=args.hours,
            seed=seed,
            start_hour=args.start_hour,
        )
        fixed_time = time.time() - t0
        fixed_summary = fixed_metrics.summary()
        fixed_summary["run_time_s"] = round(fixed_time, 2)
        fixed_results.append(fixed_summary)
        print(f"       Avg Wait: {fixed_summary['avg_wait_time']}s, "
              f"Throughput: {fixed_summary['total_throughput']} veh, "
              f"Compute: {fixed_time:.1f}s")

    # ── Aggregate Results ───────────────────────────────────────────
    def avg(results, key):
        vals = [r[key] for r in results]
        return round(np.mean(vals), 2)

    def std(results, key):
        vals = [r[key] for r in results]
        return round(np.std(vals), 2)

    summary_text = f"""
{'=' * 60}
  RESULTS SUMMARY  ({args.runs} runs × {args.hours}h each)
{'=' * 60}

  Metric              | GA Controller       | Fixed-Time
  --------------------|---------------------|-------------------
  Avg Wait Time (s)   | {avg(ga_results, 'avg_wait_time'):>7} ± {std(ga_results, 'avg_wait_time'):<6} | {avg(fixed_results, 'avg_wait_time'):>7} ± {std(fixed_results, 'avg_wait_time'):<6}
  Avg Queue Length     | {avg(ga_results, 'avg_queue_length'):>7} ± {std(ga_results, 'avg_queue_length'):<6} | {avg(fixed_results, 'avg_queue_length'):>7} ± {std(fixed_results, 'avg_queue_length'):<6}
  Total Throughput     | {avg(ga_results, 'total_throughput'):>7} ± {std(ga_results, 'total_throughput'):<6} | {avg(fixed_results, 'total_throughput'):>7} ± {std(fixed_results, 'total_throughput'):<6}

  Improvement (Wait Time): {((avg(fixed_results, 'avg_wait_time') - avg(ga_results, 'avg_wait_time')) / max(avg(fixed_results, 'avg_wait_time'), 0.01) * 100):+.1f}%
"""
    print(summary_text)

    with open(results_dir / "results_summary.txt", "w") as f:
        f.write(summary_text)

    # ── Generate Graphs ─────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-darkgrid")
    fig_w, fig_h = 10, 6

    # 1. Average Wait Time Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    labels = ["GA Controller", "Fixed-Time"]
    means = [avg(ga_results, "avg_wait_time"), avg(fixed_results, "avg_wait_time")]
    stds = [std(ga_results, "avg_wait_time"), std(fixed_results, "avg_wait_time")]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, means, yerr=stds, capsize=10, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Average Wait Time (seconds)", fontsize=12)
    ax.set_title("GA vs Fixed-Time: Average Vehicle Wait Time", fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.1f s", padding=5, fontsize=11)
    plt.tight_layout()
    plt.savefig(results_dir / "avg_wait_time_comparison.png", dpi=150)
    plt.close()

    # 2. Average Queue Length Comparison
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    means = [avg(ga_results, "avg_queue_length"), avg(fixed_results, "avg_queue_length")]
    stds = [std(ga_results, "avg_queue_length"), std(fixed_results, "avg_queue_length")]
    bars = ax.bar(labels, means, yerr=stds, capsize=10, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Average Queue Length (vehicles)", fontsize=12)
    ax.set_title("GA vs Fixed-Time: Average Queue Length", fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=11)
    plt.tight_layout()
    plt.savefig(results_dir / "avg_queue_length_comparison.png", dpi=150)
    plt.close()

    # 3. Throughput Comparison
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    means = [avg(ga_results, "total_throughput"), avg(fixed_results, "total_throughput")]
    stds = [std(ga_results, "total_throughput"), std(fixed_results, "total_throughput")]
    bars = ax.bar(labels, means, yerr=stds, capsize=10, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Total Throughput (vehicles)", fontsize=12)
    ax.set_title("GA vs Fixed-Time: Total Vehicles Cleared", fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.0f", padding=5, fontsize=11)
    plt.tight_layout()
    plt.savefig(results_dir / "throughput_comparison.png", dpi=150)
    plt.close()

    print(f"\n[DONE] Graphs saved to: {results_dir.resolve()}")
    print(f"       - avg_wait_time_comparison.png")
    print(f"       - avg_queue_length_comparison.png")
    print(f"       - throughput_comparison.png")
    print(f"       - results_summary.txt")


if __name__ == "__main__":
    main()
