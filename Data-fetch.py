import fastf1 as ff1
import pandas as pd
from pathlib import Path
from datetime import datetime


Path('cache').mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache('cache/')

def fetch_monaco(year:int):
    session_types = ['FP1', 'FP2', 'FP3', 'Q']
    dfs = {}
    for sess in session_types:
        print(f"  ↳ Downloading {year} Monaco {sess}…")
        session = ff1.get_session(year, 'Monaco', sess)
        session.load()
        laps = session.laps
        laps["Session"] = sess
        dfs[sess] = laps
    return pd.concat(dfs.values(), ignore_index=True)


if __name__ == "__main__":
    out_dir = Path('data/monaco')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the last TWO Monaco events (e.g. 2023 & 2024)
    for y in [2023, 2024]:
        df = fetch_monaco(y)
        fn = out_dir / f"monaco_{y}.parquet"
        df.to_parquet(fn)
        print(f"Saved raw data to {fn}")

    print("Done fetching Monaco practice + qualifying data.")
