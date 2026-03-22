"""
Main Entry Point — Evolutionary Traffic Signal Control
=======================================================
Ties together the simulation engine, AI controllers, data pipeline,
metrics collection, and Flask API into a single runnable application.

Architecture:
    Thread 1 (daemon)  → simulation_loop()   runs the fixed-timestep sim
    Thread 2 (main)    → Flask app           serves /state, /metrics, etc.

Usage:
    python main.py --mode ga          # Run with GA controller
    python main.py --mode fixed       # Run with Fixed-Time controller
    python main.py --mode ga --hour 8 # Start at 8 AM (peak hour)
"""

from __future__ import annotations

import argparse
import copy
import threading
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.intersection import Intersection
from core.ga_controller import GAController, GAConfig
from core.fixed_time_controller import FixedTimeController
from core.vehicle_physics import YELLOW_DURATION
from data_pipeline.traffic_generator import TrafficGenerator
from data_pipeline.rfid_handler import RFIDHandler
from metrics.collector import MetricsCollector
from server.api import create_app, init_api


# ── Configuration ───────────────────────────────────────────────────────────

TICK_RATE_MS        = 100       # 100 ms → 10 ticks per second
TICK_DURATION       = TICK_RATE_MS / 1000.0  # 0.1 seconds
GA_EVOLVE_INTERVAL  = 60.0     # Re-evolve GA every 60 simulated seconds
EMERGENCY_CHECK_INTERVAL = 5.0 # Check for RFID events every 5 sim-seconds
WARM_UP_TICKS       = 100      # Skip first 100 ticks for metrics (10s warm-up)
SIM_SPEED_MULTIPLIER = 10      # 1 tick = 1 simulated second (speed up 10x)
FLASK_PORT          = 5000


# ── Shared State ────────────────────────────────────────────────────────────

state_lock = threading.Lock()
SIMULATION_STATE: dict = {
    "timestamp": 0,
    "sim_time_seconds": 0.0,
    "sim_hour": 8,
    "lights": {"NS": "GREEN", "EW": "RED"},
    "queues": {"N": 0, "S": 0, "E": 0, "W": 0},
    "emergency_active": False,
    "emergency_direction": None,
    "controller_mode": "ga",
    "current_timings": {"ns_green": 30.0, "ew_green": 30.0},
    "stats": {
        "total_wait_vs": 0.0,
        "total_cleared": 0,
        "total_arrived": 0,
        "avg_wait": 0.0,
    },
}


# ── Simulation Loop ────────────────────────────────────────────────────────

def simulation_loop(
    intersection: Intersection,
    controller,
    controller_mode: str,
    traffic_gen: TrafficGenerator,
    rfid_handler: RFIDHandler,
    metrics: MetricsCollector,
    start_hour: int,
) -> None:
    """
    Main simulation loop.  Runs continuously in a daemon thread
    using a fixed timestep to ensure consistent simulation speed.
    """
    global SIMULATION_STATE

    tick_sec = TICK_DURATION
    sim_time = 0.0          # Cumulative simulated seconds
    last_ga_evolve = 0.0    # When we last ran the GA
    last_emergency_check = 0.0

    print(f"[SIM] Engine started — mode={controller_mode}, start_hour={start_hour}")
    print(f"[SIM] Tick rate={TICK_RATE_MS}ms, GA interval={GA_EVOLVE_INTERVAL}s")

    while True:
        loop_start = time.time()

        # Each tick advances the simulation by SIM_SPEED_MULTIPLIER seconds
        sim_step = tick_sec * SIM_SPEED_MULTIPLIER
        sim_time += sim_step

        # Current simulated hour (wraps at 24)
        current_hour = (start_hour + int(sim_time / 3600)) % 24

        # ─── 1. GENERATE TRAFFIC ARRIVALS ───────────────────────────
        arrivals = traffic_gen.get_arrivals(current_hour, sim_step)
        intersection.add_vehicles(arrivals)

        # ─── 2. CHECK FOR EMERGENCY VEHICLES ────────────────────────
        if sim_time - last_emergency_check >= EMERGENCY_CHECK_INTERVAL:
            last_emergency_check = sim_time

            if not intersection.emergency_active:
                is_emergency, direction = rfid_handler.check_emergency(
                    intersection.queues
                )
                if is_emergency and direction:
                    intersection.activate_emergency(
                        direction,
                        rfid_handler.get_override_duration(),
                    )
                    print(
                        f"[SIM] 🚑 Emergency on {direction}! "
                        f"Override for {rfid_handler.get_override_duration()}s"
                    )

        # ─── 3. EVOLVE GA (periodic) ────────────────────────────────
        if controller_mode == "ga" and not intersection.emergency_active:
            if sim_time - last_ga_evolve >= GA_EVOLVE_INTERVAL:
                last_ga_evolve = sim_time
                ns_green, ew_green = controller.evolve(intersection.queues)
                intersection.set_timings(ns_green, ew_green)
                print(
                    f"[GA]  Evolved → NS={ns_green}s, EW={ew_green}s "
                    f"(queues: {dict(intersection.queues)})"
                )

        # ─── 4. TICK THE INTERSECTION ────────────────────────────────
        state_snapshot = intersection.tick()

        # ─── 5. RECORD METRICS (skip warm-up) ───────────────────────
        if intersection.tick_count > WARM_UP_TICKS:
            metrics.record(state_snapshot)

        # ─── 6. UPDATE SHARED STATE (thread-safe) ───────────────────
        ns, ew = controller.get_current_timings()
        new_state = {
            "timestamp": intersection.tick_count,
            "sim_time_seconds": round(sim_time, 1),
            "sim_hour": current_hour,
            "lights": state_snapshot["lights"],
            "queues": state_snapshot["queues"],
            "emergency_active": state_snapshot["emergency_active"],
            "emergency_direction": state_snapshot["emergency_direction"],
            "controller_mode": controller_mode,
            "current_timings": {"ns_green": ns, "ew_green": ew},
            "stats": state_snapshot["stats"],
        }

        with state_lock:
            SIMULATION_STATE.clear()
            SIMULATION_STATE.update(new_state)

        # ─── FIXED TIMESTEP SYNC ────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_time = tick_sec - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── CLI & Entry Point ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolutionary Traffic Signal Control — Backend Server"
    )
    parser.add_argument(
        "--mode",
        choices=["ga", "fixed"],
        default="ga",
        help="Controller mode: 'ga' for Genetic Algorithm, 'fixed' for baseline",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=8,
        help="Starting hour of day (0–23). Default: 8 (morning peak)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=FLASK_PORT,
        help=f"Flask server port. Default: {FLASK_PORT}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--traffic-csv",
        type=str,
        default=None,
        help="Path to Metro Interstate Traffic Volume CSV",
    )
    parser.add_argument(
        "--rfid-csv",
        type=str,
        default=None,
        help="Path to Traffic Signal Control (RFID) CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Initialise Components ───────────────────────────────────────
    print("=" * 60)
    print("  Evolutionary Traffic Signal Control")
    print("=" * 60)
    print(f"  Mode       : {args.mode.upper()}")
    print(f"  Start Hour : {args.hour}:00")
    print(f"  Seed       : {args.seed}")
    print(f"  API Port   : {args.port}")
    print("=" * 60)

    # Data pipeline
    traffic_gen = TrafficGenerator(
        csv_path=args.traffic_csv,
        random_seed=args.seed,
    )
    rfid_handler = RFIDHandler(csv_path=args.rfid_csv)

    # Intersection
    intersection = Intersection(
        ns_green_duration=30.0,
        ew_green_duration=30.0,
        tick_duration=TICK_DURATION * SIM_SPEED_MULTIPLIER,
    )

    # Controller
    if args.mode == "ga":
        controller = GAController(
            config=GAConfig(random_seed=args.seed)
        )
    else:
        controller = FixedTimeController(ns_green=30.0, ew_green=30.0)

    # Metrics
    metrics = MetricsCollector()

    # ── Set Up API ──────────────────────────────────────────────────
    app = create_app()
    init_api(
        shared_state=SIMULATION_STATE,
        state_lock=state_lock,
        metrics_collector=metrics,
        controller=controller,
        controller_mode=args.mode,
    )

    # ── Start Simulation Thread ─────────────────────────────────────
    sim_thread = threading.Thread(
        target=simulation_loop,
        args=(
            intersection,
            controller,
            args.mode,
            traffic_gen,
            rfid_handler,
            metrics,
            args.hour,
        ),
        daemon=True,
    )
    sim_thread.start()

    # ── Start Flask Server ──────────────────────────────────────────
    print(f"\n[API] Server starting on http://127.0.0.1:{args.port}")
    print(f"[API] Endpoints:")
    print(f"       GET /state       → live intersection state")
    print(f"       GET /metrics     → performance summary")
    print(f"       GET /metrics/queues → queue history (last 60s)")
    print(f"       GET /config      → controller info")
    print(f"       GET /history     → GA evolution log\n")

    # debug=False is critical — Flask's auto-reloader spawns
    # duplicate simulation threads otherwise
    app.run(
        host="127.0.0.1",
        port=args.port,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
