# features.py
import pandas as pd
from pathlib import Path

def load_monaco_data(years, data_dir="data/monaco"):
    """Load & concatenate Monaco parquet files for given years, adding a Year column."""
    dfs = []
    for y in years:
        fn = Path(data_dir) / f"monaco_{y}.parquet"
        df = pd.read_parquet(fn)
        df['Year'] = y                       # ← annotate with year!
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compute_best_and_gap(df):
    """
    For each (Year, Session, Driver):
      - find the driver’s best lap (in seconds)
      - find the session’s overall best lap
      - compute GapToBest = BestLap – SessionBest
    """
    # convert LapTime to float seconds
    df['LapSeconds'] = df['LapTime'].dt.total_seconds()

    # Best lap per Year–Session–Driver
    best = (
        df
        .groupby(['Year','Session','Driver'], as_index=False)
        .LapSeconds
        .min()
        .rename(columns={'LapSeconds':'BestLap'})
    )

    # Best lap per Year–Session overall
    session_best = (
        best
        .groupby(['Year','Session'], as_index=False)
        .BestLap
        .min()
        .rename(columns={'BestLap':'SessionBest'})
    )

    # merge to compute the gap
    feat = best.merge(session_best, on=['Year','Session'])
    feat['GapToBest'] = feat['BestLap'] - feat['SessionBest']
    return feat

if __name__ == "__main__":
    years = [2023, 2024]
    df    = load_monaco_data(years)
    feats = compute_best_and_gap(df)

    # confirm you have 160 rows (2 years × 4 sessions × 20 drivers)
    print("Output shape:", feats.shape)
    print(feats.to_string(index=False))
