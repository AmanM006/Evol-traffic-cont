"""
Vehicle Physics Module
======================
Constants and utility functions for traffic discharge modeling at a
signalised intersection.

Key parameter:
    SATURATION_FLOW_RATE — vehicles that can clear the stop-line per second
    of effective green time.  A headway of 2 s per vehicle gives 0.5 veh/s,
    which is a realistic value for a single-lane urban approach.
"""

# ── Constants ───────────────────────────────────────────────────────────────

SATURATION_FLOW_RATE = 0.5      # vehicles per second of green (1 car / 2 s)
YELLOW_DURATION      = 4        # seconds of amber between phases
MIN_GREEN            = 10       # minimum green time in seconds
MAX_GREEN            = 60       # maximum green time in seconds
MAX_CYCLE_LENGTH     = 120      # max total cycle length in seconds


def vehicles_discharged(green_elapsed_seconds: float) -> float:
    """
    How many vehicles have been discharged cumulatively after
    *green_elapsed_seconds* of green time on a single approach.

    Uses constant saturation flow (no start-up lost time for simplicity).

    Parameters
    ----------
    green_elapsed_seconds : float
        Seconds of effective green elapsed so far in the current phase.

    Returns
    -------
    float
        Cumulative vehicles discharged (fractional; caller should floor/round).
    """
    return SATURATION_FLOW_RATE * green_elapsed_seconds


def discharge_per_tick(tick_duration: float) -> float:
    """
    Vehicles that can leave the queue in one simulation tick,
    assuming the light is green.

    Parameters
    ----------
    tick_duration : float
        Length of one simulation tick in seconds.

    Returns
    -------
    float
        Fractional vehicles that clear in one tick.
    """
    return SATURATION_FLOW_RATE * tick_duration


def wait_time_contribution(queue_length: int) -> int:
    """
    Each vehicle in the queue accumulates 1 second of waiting per
    second it remains queued.  Over one tick every queued vehicle
    adds *queue_length* vehicle-seconds of delay.

    Parameters
    ----------
    queue_length : int
        Number of vehicles currently waiting.

    Returns
    -------
    int
        Vehicle-seconds of delay generated this tick.
    """
    return queue_length
