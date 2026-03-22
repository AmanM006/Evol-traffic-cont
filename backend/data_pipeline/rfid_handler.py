"""
RFID Emergency Vehicle Handler
================================
Handles emergency / priority vehicle events from the Traffic Signal
Control Dataset for Four-Way Intersections.

When an RFID flag is detected, the handler signals which direction
should receive an immediate green override.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# How long to force green during an emergency override (seconds)
EMERGENCY_OVERRIDE_DURATION = 15.0


class RFIDHandler:
    """
    Manages emergency vehicle events from the 4-way intersection dataset.
    """

    # Updated mapping to exactly match your dataset's columns
    _ROAD_MAP = {
        "road : 01": "N",
        "road : 02": "S",
        "road : 03": "E",
        "road : 04": "W",
    }

    def __init__(
        self,
        csv_path: str | Path | None = None,
        random_probability: float = 0.005,
    ):
        self.override_duration = EMERGENCY_OVERRIDE_DURATION
        self.random_probability = random_probability

        self._events: list[dict] = []
        self._event_index: int = 0
        self._use_random: bool = True

        if csv_path and Path(csv_path).exists():
            self._load_csv(csv_path)

    def check_emergency(
        self,
        current_queues: dict[str, int],
        rng=None,
    ) -> tuple[bool, Optional[str]]:
        """Check whether an emergency vehicle event should trigger."""
        if self._use_random:
            return self._check_random(rng)
        else:
            return self._check_dataset(current_queues)

    def get_override_duration(self) -> float:
        """Return the configured override duration in seconds."""
        return self.override_duration

    # ── Internals ───────────────────────────────────────────────────────

    def _load_csv(self, file_path: str | Path) -> None:
        """Load the Traffic Signal Control dataset and extract RFID events."""
        path_str = str(file_path).lower()
        
        if path_str.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
            
        df.columns = df.columns.str.strip().str.lower()

        required = {"rfid signal"}
        if not required.issubset(set(df.columns)):
            raise ValueError(
                f"File must contain 'rfid signal' column. Found: {list(df.columns)}"
            )

        # --- NEW PROBABILITY MATH ---
        total_rows = len(df)
        emergency_rows = df[df["rfid signal"] == 1]
        
        # Calculate dataset probability and scale it for the demo
        # (e.g., capping it so you get an ambulance roughly every 2-3 minutes)
        raw_prob = len(emergency_rows) / total_rows if total_rows > 0 else 0.05
        self.trigger_probability = min(raw_prob * 0.2, 0.02) 

        for _, row in emergency_rows.iterrows():
            # Find which road has the highest density → emergency vehicle direction
            road_densities = {}
            for col, direction in self._ROAD_MAP.items():
                if col in row.index:
                    road_densities[direction] = row[col]

            if road_densities:
                priority_dir = max(road_densities, key=road_densities.get)
            else:
                priority_dir = "N"  # fallback

            self._events.append({
                "direction": priority_dir,
                "densities": road_densities,
            })

        if self._events:
            self._use_random = False

    def _check_dataset(self, current_queues: dict[str, int]) -> tuple[bool, Optional[str]]:
        """Cycle through dataset emergency events probabilistically."""
        import random
        if not self._events:
            return (False, None)

        # Roll the dice against the historical probability!
        if random.random() < self.trigger_probability:
            event = self._events[self._event_index % len(self._events)]
            self._event_index += 1
            return (True, event["direction"])

        return (False, None)

    def _check_random(self, rng=None) -> tuple[bool, Optional[str]]:
        """Generate random emergency events with configured probability."""
        import random as stdlib_random

        if rng is not None:
            roll = rng.random()
        else:
            roll = stdlib_random.random()

        if roll < self.random_probability:
            direction = stdlib_random.choice(["N", "S", "E", "W"])
            return (True, direction)

        return (False, None)