import pandas as pd
from pathlib import Path
import numpy as np
from elo_ratings import compute_driver_elo
from elo_ratings import fetch_all_results

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

def pivot_practice_with_target(feats):
    """For TRAINING: pivot FP1/FP2 → Best_FP1, Best_FP2, Delta_FP2_FP1 plus Best_Q."""
    # keep only practice & quali
    prac = feats[feats.Session.isin(['FP1','FP2'])]
    quali = feats[feats.Session=='Q'][['Year','Driver','BestLap']].rename(columns={'BestLap':'Best_Q'})

    # pivot practice bests
    wide_prac = (
      prac
      .pivot(index=['Year','Driver'], columns='Session', values='BestLap')
      .rename(columns={'FP1':'Best_FP1','FP2':'Best_FP2'})
      .reset_index()
    )
    # delta feature
    wide_prac['Delta_FP2_FP1'] = wide_prac.Best_FP2 - wide_prac.Best_FP1

    # merge in training target
    return wide_prac.merge(quali, on=['Year','Driver'], how='inner')

def pivot_practice_only(feats, year):
    prac = feats[
        (feats.Session.isin(['FP1','FP2'])) &
        (feats.Year == year)
    ]
    wide = (
        prac
        .pivot(index=['Year','Driver'], columns='Session', values='BestLap')
        .rename(columns={'FP1':'Best_FP1','FP2':'Best_FP2'})
        .reset_index()
    )
    wide['Delta_FP2_FP1'] = wide.Best_FP2 - wide.Best_FP1
    return wide

# def pivot_and_compute_deltas(feats):
#     """
#     Input: feats DataFrame with columns [Year, Session, Driver, BestLap].
#     Output: wide DataFrame with one row per (Year, Driver):
#       Best_FP1, Best_FP2, Best_FP3, Best_Q, Delta_Q_FP3
#     """
#     # pivot BestLap into separate Session columns
#     wide = (
#         feats
#         .pivot(index=['Year','Driver'], columns='Session', values='BestLap')
#         .reset_index()
#         .rename(columns={
#             'FP1':'Best_FP1',
#             'FP2':'Best_FP2',
#             'FP3':'Best_FP3',
#             'Q':'Best_Q'
#         })
#     )

#     # compute qualifying vs FP3 delta
#     wide['Delta_Q_FP3'] = wide['Best_Q'] - wide['Best_FP3']
#     return wide

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

def enrich_with_elo(wide, years):
    # 1) compute driver Elo
    driver_elo = compute_driver_elo(years)
    wide = wide.merge(driver_elo, on=['Year','Driver'], how='left')

    # 2) map drivers to teams (you might already have this in your results df)
    team_map = (
      fetch_all_results(years)[['Year','Driver','Team']]
      .drop_duplicates()
    )
    wide = wide.merge(team_map, on=['Year','Driver'], how='left')

    # 3) compute team‐level Elo
    team_elo = (
      wide
      .groupby(['Year','Team'], as_index=False)['Elo']
      .mean()
      .rename(columns={'Elo':'TeamElo'})
    )
    wide = wide.merge(team_elo, on=['Year','Team'], how='left')
    return wide

if __name__ == "__main__":
    hist_years = [2023, 2024]
    up_year    = 2025

    # 1) load & gap‐compute everything
    df_all = load_monaco_data(hist_years + [up_year])
    feats  = compute_best_and_gap(df_all)

    # 2) build train & upcoming tables
    train_df = pivot_practice_with_target(feats[feats.Year.isin(hist_years)])
    up_df    = pivot_practice_only   (feats, year=up_year)

    # 3) enrich both with Elo
    train_df = enrich_with_elo(train_df, hist_years)
    up_df    = enrich_with_elo(up_df,    hist_years)  # note: use hist Elo for rookies

    # 4) recency‐weight (only if you plan to use as sample_weight)
    train_df = apply_recency_weight(train_df)

    # 5) filter out any drivers in train_df not present this year
    current = set(up_df.Driver)
    train_df = train_df[train_df.Driver.isin(current)].reset_index(drop=True)

    # 6) save your CSVs
    train_df.to_csv('features_train.csv',    index=False)
    up_df   .to_csv('features_upcoming.csv', index=False)

    print("Train rows:", train_df.shape)
    print("Upcoming rows:", up_df.shape)
