"""
Traffic Generator (Poisson Spawner)
====================================
Converts hourly traffic volumes from the Traffic Prediction Dataset
into per-tick vehicle arrivals using a Poisson process.

Data Mapping:
    Junction 1 -> North
    Junction 2 -> South
    Junction 3 -> East
    Junction 4 -> West
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class TrafficGenerator:
    """
    Generates per-tick vehicle arrivals from Kaggle hourly traffic volumes.
    """

    def __init__(
        self,
        csv_path: str | Path | None = None,
        random_seed: int | None = None,
    ):
        self.rng = np.random.default_rng(random_seed)

        # Dictionary structure: {hour: {"N": vol, "S": vol, "E": vol, "W": vol}}
        self.hourly_volumes: dict[int, dict[str, float]] = {}
        
        if csv_path and Path(csv_path).exists():
            self._load_csv(csv_path)
        else:
            self._generate_synthetic_volumes()

    def get_arrivals(
        self,
        hour: int,
        tick_duration: float,
    ) -> dict[str, int]:
        """Generate Poisson-distributed vehicle arrivals for one tick."""
        
        vols = self.hourly_volumes.get(hour % 24, {"N": 350, "S": 350, "E": 150, "W": 150})

        # --- DEMO MULTIPLIER ---
        # Multiply the Kaggle volume by 15 so the intersection gets packed quickly
        DEMO_VOLUME_MULTIPLIER = 15.0 

        arrivals = {}
        for direction in ["N", "S", "E", "W"]:
            # λ = (hourly_volume * multiplier) / 3600 * tick_duration
            lam = (vols[direction] * DEMO_VOLUME_MULTIPLIER / 3600.0) * tick_duration
            arrivals[direction] = int(self.rng.poisson(lam))

        return arrivals
    # ── Internals ───────────────────────────────────────────────────────

    def _load_csv(self, csv_path: str | Path) -> None:
        """
        Load the Traffic Prediction CSV.
        Expected columns: ['datetime', 'junction', 'vehicles', 'id']
        """
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()

        if "vehicles" not in df.columns or "junction" not in df.columns:
            raise ValueError(
                f"CSV must contain 'vehicles' and 'junction' columns. "
                f"Found columns: {list(df.columns)}"
            )

        # Extract hour from datetime column
        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

        # Average the vehicles per hour per junction across the whole dataset
        hourly_junction_avg = df.groupby(["hour", "junction"])["vehicles"].mean().reset_index()

        # Map Kaggle junctions to our simulation directions
        j_map = {1: "N", 2: "S", 3: "E", 4: "W"}

        # Initialize the dictionary with 0s
        for h in range(24):
            self.hourly_volumes[h] = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}

        # Populate the dictionary with real data
        for _, row in hourly_junction_avg.iterrows():
            h = int(row["hour"])
            j = int(row["junction"])
            v = float(row["vehicles"])
            
            if j in j_map:
                self.hourly_volumes[h][j_map[j]] = v

    def _generate_synthetic_volumes(self) -> None:
        """Fallback synthetic data if CSV fails to load."""
        for h in range(24):
            # Peak hours get 1000 cars base, off-peak get 200
            base = 1000 if 7 <= h <= 19 else 200
            self.hourly_volumes[h] = {
                "N": base * 0.35, 
                "S": base * 0.35, 
                "E": base * 0.15, 
                "W": base * 0.15
            }