"""
Fixed-Time Signal Controller (Baseline)
========================================
A simple controller that uses static, pre-configured green durations
for each phase.  This serves as the baseline against which the GA
controller is compared.
"""

from core.vehicle_physics import MIN_GREEN, MAX_GREEN


class FixedTimeController:
    """
    Returns the same green durations every time — no adaptation.

    Parameters
    ----------
    ns_green : float
        Green duration for North/South phase (seconds).
    ew_green : float
        Green duration for East/West phase (seconds).
    """

    def __init__(self, ns_green: float = 30.0, ew_green: float = 30.0):
        self.ns_green = max(MIN_GREEN, min(MAX_GREEN, ns_green))
        self.ew_green = max(MIN_GREEN, min(MAX_GREEN, ew_green))

    def get_current_timings(self) -> tuple[float, float]:
        """Return the fixed (NS_green, EW_green) durations."""
        return (self.ns_green, self.ew_green)

    def __repr__(self) -> str:
        return f"FixedTimeController(ns={self.ns_green}s, ew={self.ew_green}s)"
