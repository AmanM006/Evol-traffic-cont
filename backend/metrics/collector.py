"""
Metrics Collector
==================
Tracks per-tick simulation performance data and provides aggregation
methods for comparison graphs between GA and Fixed-Time controllers.

Logged per tick:
    - timestamp (tick number)
    - queue lengths per direction
    - total queue length
    - vehicles cleared this tick
    - cumulative wait time (vehicle-seconds)
    - current light phase
    - whether an emergency is active
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TickRecord:
    """One row of per-tick metrics."""
    tick: int
    queues: dict[str, int]
    total_queue: int
    vehicles_cleared: int
    cumulative_wait_vs: float
    phase: str
    emergency_active: bool


class MetricsCollector:
    """
    Accumulates per-tick simulation data and provides summary statistics.
    """

    def __init__(self):
        self.records: list[TickRecord] = []
        self._prev_cleared: int = 0  # to compute per-tick delta

    def record(self, state: dict) -> None:
        """
        Record a tick from an intersection state snapshot.

        Parameters
        ----------
        state : dict
            The dict returned by ``Intersection.get_state()``.
        """
        current_cleared = state["stats"]["total_cleared"]
        cleared_this_tick = current_cleared - self._prev_cleared
        self._prev_cleared = current_cleared

        record = TickRecord(
            tick=state["tick"],
            queues=dict(state["queues"]),
            total_queue=sum(state["queues"].values()),
            vehicles_cleared=cleared_this_tick,
            cumulative_wait_vs=state["stats"]["total_wait_vs"],
            phase=state["phase"],
            emergency_active=state["emergency_active"],
        )
        self.records.append(record)

    def avg_wait_time(self) -> float:
        """
        Average wait time per vehicle over the entire recorded period.
        Uses the last recorded cumulative wait divided by total cleared.
        """
        if not self.records:
            return 0.0
        last = self.records[-1]
        total_cleared = sum(r.vehicles_cleared for r in self.records)
        if total_cleared == 0:
            return 0.0
        return round(last.cumulative_wait_vs / total_cleared, 2)

    def avg_queue_length(self) -> float:
        """Average total queue length across all recorded ticks."""
        if not self.records:
            return 0.0
        return round(
            sum(r.total_queue for r in self.records) / len(self.records), 2
        )

    def total_throughput(self) -> int:
        """Total vehicles cleared over the recorded period."""
        return sum(r.vehicles_cleared for r in self.records)

    def throughput_per_minute(self, tick_duration: float) -> float:
        """Vehicles cleared per simulated minute."""
        if not self.records:
            return 0.0
        total_sim_seconds = len(self.records) * tick_duration
        if total_sim_seconds == 0:
            return 0.0
        total = self.total_throughput()
        return round(total / (total_sim_seconds / 60.0), 2)

    def queue_length_over_time(self) -> list[dict]:
        """
        Return queue lengths over time for graphing.

        Returns
        -------
        list[dict]
            Each dict has 'tick', 'N', 'S', 'E', 'W', 'total'.
        """
        return [
            {
                "tick": r.tick,
                **r.queues,
                "total": r.total_queue,
            }
            for r in self.records
        ]

    def summary(self) -> dict:
        """Return a summary dict suitable for JSON serialisation."""
        return {
            "avg_wait_time": self.avg_wait_time(),
            "avg_queue_length": self.avg_queue_length(),
            "total_throughput": self.total_throughput(),
            "ticks_recorded": len(self.records),
        }

    def export_data(self) -> list[dict]:
        """Export all records as a list of dicts (for CSV/DataFrame)."""
        return [
            {
                "tick": r.tick,
                "total_queue": r.total_queue,
                "vehicles_cleared": r.vehicles_cleared,
                "cumulative_wait_vs": r.cumulative_wait_vs,
                "phase": r.phase,
                "emergency_active": r.emergency_active,
                **{f"queue_{d}": v for d, v in r.queues.items()},
            }
            for r in self.records
        ]

    def reset(self) -> None:
        """Clear all recorded data (for between-experiment cleanup)."""
        self.records.clear()
        self._prev_cleared = 0
