import pandas as pd
from pathlib import Path
import numpy as np

def load_monaco_data(years, data_dir="data/monaco"):
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

def pivot_and_compute_deltas(feats):
    """
    Input: feats DataFrame with columns [Year, Session, Driver, BestLap].
    Output: wide DataFrame with one row per (Year, Driver):
      Best_FP1, Best_FP2, Best_FP3, Best_Q, Delta_Q_FP3
    """
    # pivot BestLap into separate Session columns
    wide = (
        feats
        .pivot(index=['Year','Driver'], columns='Session', values='BestLap')
        .reset_index()
        .rename(columns={
            'FP1':'Best_FP1',
            'FP2':'Best_FP2',
            'FP3':'Best_FP3',
            'Q':'Best_Q'
        })
    )

    # compute qualifying vs FP3 delta
    wide['Delta_Q_FP3'] = wide['Best_Q'] - wide['Best_FP3']
    return wide

#Adding Recency weights
def apply_recency_weight(wide, decay=1.0):
    """
    Adds RecencyWeight = exp(–decay*(max_year – Year))
    so the most recent year has weight = 1.0.
    Basically, last year's data will show that Leclerc topped all the sessions, with Mclaren behind, the year prio Redbull topped with Aston right behind but that order has completely shaken up.
    Also need to figure out a way to factor in this year's car pace data from the races so far.
    """
    max_year = wide['Year'].max()
    wide['RecencyWeight'] = np.exp(-decay*(max_year - wide['Year']))
    return wide



if __name__ == "__main__":
    years = [2023, 2024]
    df    = load_monaco_data(years)
    feats = compute_best_and_gap(df)
    wide = pivot_and_compute_deltas(feats)
    wide = apply_recency_weight(wide, decay=1.0)

    print("Shape:", wide.shape)  # should be (40, 6) → 2 years × ~20 drivers × 6 cols
    print(wide.to_string(index=False))

