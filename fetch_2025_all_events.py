import pandas as pd
from pathlib import Path
import fastf1
from fastf1_data import fetch_event_sessions


def fetch_all_2025_races(sessions=("FP1", "FP2", "FP3", "Q", "R")):
    """Fetch lap data for every 2025 Grand Prix and save to parquet."""
    schedule = fastf1.get_event_schedule(2025)
    out_dir = Path("data/2025")
    out_dir.mkdir(parents=True, exist_ok=True)

    for event in schedule["EventName"]:
        print(f"\nFetching sessions for {event} 2025")
        df = fetch_event_sessions(2025, event, sessions=sessions)
        fname = event.lower().replace(" ", "_")
        fn = out_dir / f"{fname}_2025.parquet"
        df.to_parquet(fn)
        print(f"✔ Saved {len(df)} laps to {fn}")


if __name__ == "__main__":
    fetch_all_2025_races()

