"""
Intersection State Machine
===========================
Models a 4-way signalised intersection with two opposing phases:

    Phase 1  — NS_GREEN  (North/South have green, East/West have red)
    Phase 2  — EW_GREEN  (East/West have green, North/South have red)

Each green phase is followed by a mandatory YELLOW transition of
`YELLOW_DURATION` seconds before the opposing phase begins.

State diagram:
    NS_GREEN ──► NS_YELLOW ──► EW_GREEN ──► EW_YELLOW ──► NS_GREEN ──► …
"""

from enum import Enum, auto
from core.vehicle_physics import (
    discharge_per_tick,
    wait_time_contribution,
    YELLOW_DURATION,
)


class Phase(Enum):
    NS_GREEN  = auto()
    NS_YELLOW = auto()
    EW_GREEN  = auto()
    EW_YELLOW = auto()


# Which directions get green in each phase
_GREEN_DIRECTIONS = {
    Phase.NS_GREEN:  ("N", "S"),
    Phase.EW_GREEN:  ("E", "W"),
}

# Phase transition order
_NEXT_PHASE = {
    Phase.NS_GREEN:  Phase.NS_YELLOW,
    Phase.NS_YELLOW: Phase.EW_GREEN,
    Phase.EW_GREEN:  Phase.EW_YELLOW,
    Phase.EW_YELLOW: Phase.NS_GREEN,
}


class Intersection:
    """
    Manages vehicle queues, signal phases, and per-tick physics for a
    four-way intersection.

    Parameters
    ----------
    ns_green_duration : float
        Initial green time (seconds) for the North/South phase.
    ew_green_duration : float
        Initial green time (seconds) for the East/West phase.
    tick_duration : float
        Length of one simulation tick in seconds (e.g. 0.1 for 100 ms).
    """

    def __init__(
        self,
        ns_green_duration: float = 30.0,
        ew_green_duration: float = 30.0,
        tick_duration: float = 0.1,
    ):
        # Signal timings (can be updated by the GA controller)
        self.ns_green_duration = ns_green_duration
        self.ew_green_duration = ew_green_duration
        self.tick_duration = tick_duration

        # Current phase and time remaining in that phase (seconds)
        self.phase = Phase.NS_GREEN
        self.phase_timer = ns_green_duration

        # Vehicle queues — one per approach
        self.queues: dict[str, int] = {"N": 0, "S": 0, "E": 0, "W": 0}

        # Fractional discharge accumulator (we only release whole vehicles)
        self._discharge_accum: dict[str, float] = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}

        # Pending timing update (applied at next cycle start)
        self._pending_ns_green: float | None = None
        self._pending_ew_green: float | None = None

        # Cumulative statistics (reset-able)
        self.total_wait_vehicle_seconds: float = 0.0
        self.total_vehicles_cleared: int = 0
        self.total_vehicles_arrived: int = 0
        self.tick_count: int = 0

        # Emergency override state
        self.emergency_active: bool = False
        self.emergency_direction: str | None = None
        self.emergency_timer: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────

    def set_timings(self, ns_green: float, ew_green: float) -> None:
        """Queue green duration update (applied at next cycle start)."""
        self._pending_ns_green = ns_green
        self._pending_ew_green = ew_green

    def add_vehicles(self, arrivals: dict[str, int]) -> None:
        """
        Add newly-arrived vehicles to the queues.

        Parameters
        ----------
        arrivals : dict
            e.g. {"N": 2, "S": 0, "E": 1, "W": 0}
        """
        for direction, count in arrivals.items():
            self.queues[direction] += count
            self.total_vehicles_arrived += count

    def activate_emergency(self, direction: str, duration: float = 15.0) -> None:
        """
        Force green for *direction* for *duration* seconds.
        The intersection immediately transitions to give that direction green.
        """
        self.emergency_active = True
        self.emergency_direction = direction
        self.emergency_timer = duration

        # Determine which phase gives this direction green
        if direction in ("N", "S"):
            self.phase = Phase.NS_GREEN
            self.phase_timer = duration
        else:
            self.phase = Phase.EW_GREEN
            self.phase_timer = duration

    def deactivate_emergency(self) -> None:
        """End the emergency override and resume normal cycling."""
        self.emergency_active = False
        self.emergency_direction = None
        self.emergency_timer = 0.0

    def tick(self) -> dict:
        """
        Advance the simulation by one tick.

        Returns
        -------
        dict
            Snapshot of the intersection state after this tick.
        """
        self.tick_count += 1

        # 1. Accumulate waiting time for ALL queued vehicles
        for direction in self.queues:
            self.total_wait_vehicle_seconds += (
                wait_time_contribution(self.queues[direction]) * self.tick_duration
            )

        # 2. Discharge vehicles if the current phase has a green direction
        green_dirs = _GREEN_DIRECTIONS.get(self.phase, ())
        per_tick = discharge_per_tick(self.tick_duration)

        for d in green_dirs:
            if self.queues[d] > 0:
                self._discharge_accum[d] += per_tick
                # Release whole vehicles
                released = int(self._discharge_accum[d])
                released = min(released, self.queues[d])  # can't release more than queued
                if released > 0:
                    self.queues[d] -= released
                    self._discharge_accum[d] -= released
                    self.total_vehicles_cleared += released

        # 3. Advance the phase timer
        self.phase_timer -= self.tick_duration

        # 4. Handle emergency override countdown
        if self.emergency_active:
            self.emergency_timer -= self.tick_duration
            if self.emergency_timer <= 0:
                self.deactivate_emergency()
                # After emergency, continue with a yellow transition
                self.phase = _NEXT_PHASE[self.phase]
                self.phase_timer = YELLOW_DURATION

        # 5. Phase transition (normal operation)
        if self.phase_timer <= 0 and not self.emergency_active:
            self._transition_phase()

        return self.get_state()

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of the current intersection."""
        return {
            "tick": self.tick_count,
            "phase": self.phase.name,
            "phase_timer": round(self.phase_timer, 2),
            "lights": self._lights_dict(),
            "queues": dict(self.queues),
            "emergency_active": self.emergency_active,
            "emergency_direction": self.emergency_direction,
            "stats": {
                "total_wait_vs": round(self.total_wait_vehicle_seconds, 2),
                "total_cleared": self.total_vehicles_cleared,
                "total_arrived": self.total_vehicles_arrived,
                "avg_wait": self._avg_wait_time(),
            },
        }

    def reset_stats(self) -> None:
        """Reset cumulative statistics (for warm-up period exclusion)."""
        self.total_wait_vehicle_seconds = 0.0
        self.total_vehicles_cleared = 0
        self.total_vehicles_arrived = 0
        self.tick_count = 0

    # ── Internals ───────────────────────────────────────────────────────

    def _transition_phase(self) -> None:
        """Move to the next phase and set the timer."""
        self.phase = _NEXT_PHASE[self.phase]

        if self.phase == Phase.NS_GREEN:
            # Apply pending timing updates at cycle boundary
            if self._pending_ns_green is not None:
                self.ns_green_duration = self._pending_ns_green
                self.ew_green_duration = self._pending_ew_green
                self._pending_ns_green = None
                self._pending_ew_green = None
            self.phase_timer = self.ns_green_duration
        elif self.phase == Phase.EW_GREEN:
            self.phase_timer = self.ew_green_duration
        else:
            # Yellow phase
            self.phase_timer = YELLOW_DURATION

        # Reset discharge accumulators on phase change
        for d in self._discharge_accum:
            self._discharge_accum[d] = 0.0

    def _lights_dict(self) -> dict[str, str]:
        """Human-readable light colours for the frontend."""
        if self.phase == Phase.NS_GREEN:
            return {"NS": "GREEN", "EW": "RED"}
        elif self.phase == Phase.NS_YELLOW:
            return {"NS": "YELLOW", "EW": "RED"}
        elif self.phase == Phase.EW_GREEN:
            return {"NS": "RED", "EW": "GREEN"}
        else:  # EW_YELLOW
            return {"NS": "RED", "EW": "YELLOW"}

    def _avg_wait_time(self) -> float:
        """Average wait time per vehicle (seconds)."""
        if self.total_vehicles_cleared == 0:
            return 0.0
        return round(self.total_wait_vehicle_seconds / self.total_vehicles_cleared, 2)
