"""
Smoke Tests — Evolutionary Traffic Signal Control Backend
==========================================================
Verifies core components work correctly in isolation.

Run:
    cd backend
    python -m pytest tests/smoke_test.py -v
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.intersection import Intersection, Phase
from core.ga_controller import GAController, GAConfig
from core.fixed_time_controller import FixedTimeController
from core.vehicle_physics import (
    MIN_GREEN, MAX_GREEN, SATURATION_FLOW_RATE, discharge_per_tick,
)
from data_pipeline.traffic_generator import TrafficGenerator
from metrics.collector import MetricsCollector


# ── Intersection Tests ──────────────────────────────────────────────────────

class TestIntersection:
    """Verify the intersection state machine and discharge logic."""

    def test_initial_state(self):
        ix = Intersection(ns_green_duration=30.0, ew_green_duration=30.0)
        assert ix.phase == Phase.NS_GREEN
        assert all(q == 0 for q in ix.queues.values())
        assert ix.tick_count == 0

    def test_add_vehicles(self):
        ix = Intersection()
        ix.add_vehicles({"N": 5, "S": 3, "E": 2, "W": 1})
        assert ix.queues == {"N": 5, "S": 3, "E": 2, "W": 1}
        assert ix.total_vehicles_arrived == 11

    def test_queues_decrease_during_green(self):
        """Run 100 ticks with NS_GREEN and verify N/S queues decrease."""
        ix = Intersection(
            ns_green_duration=30.0,
            ew_green_duration=30.0,
            tick_duration=1.0,  # 1 second ticks for faster discharge
        )
        ix.queues = {"N": 20, "S": 15, "E": 10, "W": 5}
        initial_n = ix.queues["N"]
        initial_s = ix.queues["S"]

        for _ in range(100):
            ix.tick()

        # N and S should have decreased (they get green in NS_GREEN phase)
        assert ix.queues["N"] < initial_n, "N queue should decrease during NS_GREEN"
        assert ix.queues["S"] < initial_s, "S queue should decrease during NS_GREEN"
        assert ix.total_vehicles_cleared > 0

    def test_phase_transitions(self):
        """Verify that phases cycle correctly."""
        ix = Intersection(
            ns_green_duration=5.0,
            ew_green_duration=5.0,
            tick_duration=1.0,
        )
        phases_seen = set()
        for _ in range(200):
            ix.tick()
            phases_seen.add(ix.phase)

        assert len(phases_seen) == 4, f"Should see all 4 phases, saw {phases_seen}"

    def test_get_state_format(self):
        """Verify the state dict has all expected keys."""
        ix = Intersection()
        ix.add_vehicles({"N": 3, "S": 2, "E": 1, "W": 0})
        state = ix.tick()

        required_keys = {"tick", "phase", "phase_timer", "lights", "queues",
                         "emergency_active", "emergency_direction", "stats"}
        assert required_keys.issubset(state.keys())
        assert set(state["lights"].keys()) == {"NS", "EW"}
        assert state["lights"]["NS"] in ("GREEN", "YELLOW", "RED")
        assert state["lights"]["EW"] in ("GREEN", "YELLOW", "RED")

    def test_emergency_override(self):
        """Verify emergency override forces the correct phase."""
        ix = Intersection()
        ix.activate_emergency("E", duration=15.0)
        assert ix.emergency_active is True
        assert ix.phase == Phase.EW_GREEN

    def test_set_timings_applied_at_cycle_boundary(self):
        """Pending timings should be applied when NS_GREEN starts."""
        ix = Intersection(
            ns_green_duration=5.0, ew_green_duration=5.0, tick_duration=1.0
        )
        ix.set_timings(40.0, 20.0)

        # Run enough ticks to go through one full cycle
        for _ in range(500):
            ix.tick()

        assert ix.ns_green_duration == 40.0
        assert ix.ew_green_duration == 20.0


# ── GA Controller Tests ─────────────────────────────────────────────────────

class TestGAController:
    """Verify the Genetic Algorithm controller."""

    def test_evolve_returns_valid_timings(self):
        ga = GAController(config=GAConfig(
            population_size=10, generations=5, random_seed=42
        ))
        queues = {"N": 10, "S": 8, "E": 3, "W": 5}
        ns, ew = ga.evolve(queues)

        assert MIN_GREEN <= ns <= MAX_GREEN, f"NS={ns} outside [{MIN_GREEN}, {MAX_GREEN}]"
        assert MIN_GREEN <= ew <= MAX_GREEN, f"EW={ew} outside [{MIN_GREEN}, {MAX_GREEN}]"

    def test_evolve_with_empty_queues(self):
        ga = GAController(config=GAConfig(
            population_size=10, generations=5, random_seed=42
        ))
        ns, ew = ga.evolve({"N": 0, "S": 0, "E": 0, "W": 0})
        assert MIN_GREEN <= ns <= MAX_GREEN
        assert MIN_GREEN <= ew <= MAX_GREEN

    def test_evolve_with_asymmetric_demand(self):
        """GA should allocate more green to the busier direction."""
        ga = GAController(config=GAConfig(
            population_size=20, generations=30, random_seed=42
        ))
        # Heavily NS-dominant traffic
        ns, ew = ga.evolve({"N": 50, "S": 40, "E": 2, "W": 1})
        assert ns > ew, f"Expected NS ({ns}) > EW ({ew}) for NS-heavy traffic"

    def test_evolution_history_tracked(self):
        ga = GAController(config=GAConfig(
            population_size=10, generations=5, random_seed=42
        ))
        ga.evolve({"N": 5, "S": 5, "E": 5, "W": 5})
        history = ga.get_evolution_history()
        assert len(history) == 1
        assert "best_fitness" in history[0]
        assert "best_ns_green" in history[0]

    def test_get_current_timings(self):
        ga = GAController()
        ns, ew = ga.get_current_timings()
        # Default is 30.0, 30.0
        assert ns == 30.0
        assert ew == 30.0


# ── Fixed-Time Controller Tests ─────────────────────────────────────────────

class TestFixedTimeController:
    """Verify the baseline fixed-time controller."""

    def test_default_timings(self):
        fc = FixedTimeController()
        ns, ew = fc.get_current_timings()
        assert ns == 30.0
        assert ew == 30.0

    def test_custom_timings(self):
        fc = FixedTimeController(ns_green=45.0, ew_green=25.0)
        ns, ew = fc.get_current_timings()
        assert ns == 45.0
        assert ew == 25.0

    def test_timings_clamped(self):
        fc = FixedTimeController(ns_green=100.0, ew_green=5.0)
        ns, ew = fc.get_current_timings()
        assert ns == MAX_GREEN
        assert ew == MIN_GREEN


# ── Traffic Generator Tests ─────────────────────────────────────────────────

class TestTrafficGenerator:
    """Verify the Poisson traffic generator."""

    def test_get_arrivals_returns_all_directions(self):
        tg = TrafficGenerator(random_seed=42)
        arrivals = tg.get_arrivals(hour=8, tick_duration=1.0)
        assert set(arrivals.keys()) == {"N", "S", "E", "W"}
        assert all(isinstance(v, int) for v in arrivals.values())
        assert all(v >= 0 for v in arrivals.values())

    def test_peak_hour_has_more_traffic(self):
        """Peak hour (8 AM) should have higher volume than night (3 AM)."""
        tg = TrafficGenerator(random_seed=42)
        assert tg.get_hourly_volume(8) > tg.get_hourly_volume(3)

    def test_synthetic_24h_profile(self):
        tg = TrafficGenerator(random_seed=42)
        volumes = tg.get_all_hourly_volumes()
        assert len(volumes) == 24
        assert all(v > 0 for v in volumes.values())


# ── Metrics Collector Tests ─────────────────────────────────────────────────

class TestMetricsCollector:
    """Verify the metrics collector."""

    def test_record_and_summary(self):
        mc = MetricsCollector()
        ix = Intersection(tick_duration=1.0)
        ix.add_vehicles({"N": 10, "S": 10, "E": 10, "W": 10})

        for _ in range(50):
            state = ix.tick()
            mc.record(state)

        summary = mc.summary()
        assert "avg_wait_time" in summary
        assert "avg_queue_length" in summary
        assert "total_throughput" in summary
        assert summary["ticks_recorded"] == 50

    def test_avg_wait_time_is_sensible(self):
        mc = MetricsCollector()
        ix = Intersection(tick_duration=1.0)
        ix.add_vehicles({"N": 10, "S": 10, "E": 5, "W": 5})

        for _ in range(100):
            state = ix.tick()
            mc.record(state)

        avg_wait = mc.avg_wait_time()
        assert avg_wait >= 0, "Average wait time should be non-negative"
        # With 30 vehicles and discharge rate of 0.5/s, should be reasonable
        assert avg_wait < 10000, "Average wait time seems unreasonably high"

    def test_queue_length_over_time(self):
        mc = MetricsCollector()
        ix = Intersection(tick_duration=1.0)
        ix.add_vehicles({"N": 5, "S": 5, "E": 5, "W": 5})

        for _ in range(10):
            state = ix.tick()
            mc.record(state)

        data = mc.queue_length_over_time()
        assert len(data) == 10
        assert "tick" in data[0]
        assert "N" in data[0]
        assert "total" in data[0]

    def test_reset(self):
        mc = MetricsCollector()
        ix = Intersection(tick_duration=1.0)
        ix.add_vehicles({"N": 5, "S": 5, "E": 5, "W": 5})

        for _ in range(10):
            mc.record(ix.tick())

        mc.reset()
        assert len(mc.records) == 0
        assert mc.total_throughput() == 0
