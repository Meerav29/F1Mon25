import fastf1 as ff1
import pandas as pd
from pathlib import Path
from datetime import datetime
from fastf1.core import DataNotLoadedError


Path('cache').mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache('cache/')

def fetch_monaco(year: int):
    dfs = []
    for sess_type in ['FP1','FP2','FP3','Q']:
        print(f"→ Attempting {year} Monaco {sess_type}…")
        session = ff1.get_session(year, 'Monaco', sess_type)

        try:
            session.load()              # try to load *any* data
            laps = session.laps         # may raise DataNotLoadedError
        except DataNotLoadedError:
            print(f"   • No lap data for {year} {sess_type}, skipping")
            continue

        if laps.empty:
            print(f"   • {year} {sess_type} returned 0 laps, skipping")
            continue

        df = laps.copy()
        df['Session'] = sess_type
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No sessions fetched for Monaco {year}")
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    out_dir = Path('data/monaco')
    out_dir.mkdir(parents=True, exist_ok=True)

    for y in [2023, 2024, 2025]:
        print(f"\nFetching all Monaco data for {y}")
        df = fetch_monaco(y)
        fn = out_dir / f"monaco_{y}.parquet"
        df.to_parquet(fn)
        print(f"✔ Saved {len(df)} laps to {fn}")
