"""Utility script to download Monaco lap data."""

from pathlib import Path
import pandas as pd

from fastf1_data import fetch_event_sessions


def fetch_monaco(year: int):
    """Convenience wrapper for fetch_event_sessions."""
    return fetch_event_sessions(year, "Monaco")

if __name__ == "__main__":
    out_dir = Path('data/monaco')
    out_dir.mkdir(parents=True, exist_ok=True)

    for y in [2023, 2024, 2025]:
        print(f"\nFetching all Monaco data for {y}")
        df = fetch_monaco(y)
        fn = out_dir / f"monaco_{y}.parquet"
        df.to_parquet(fn)
        print(f"✔ Saved {len(df)} laps to {fn}")
