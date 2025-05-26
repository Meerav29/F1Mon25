# add_fp3_and_gap.py
import pandas as pd
from pathlib import Path

def load_best_fp3(year):
    """Read the Monaco parquet for `year`, extract each driver’s best FP3 lap."""
    fn = Path('data/monaco') / f'monaco_{year}.parquet'
    df = pd.read_parquet(fn)
    df['Year'] = year
    df['LapSeconds'] = df.LapTime.dt.total_seconds()
    return (
        df[df.Session == 'FP3']
          .groupby(['Year','Driver'], as_index=False)['LapSeconds']
          .min()
          .rename(columns={'LapSeconds':'Best_FP3'})
    )

# 1) Load your *base* feature tables (which have FP1, FP2 but no FP3)
train = pd.read_csv('features_train.csv')
up    = pd.read_csv('features_upcoming.csv')

# 2) Merge in Best_FP3 for training years
years = sorted(train.Year.unique())
best_fp3_train = pd.concat([load_best_fp3(y) for y in years],
                           ignore_index=True)
train = train.merge(best_fp3_train, on=['Year','Driver'], how='left')

# 3) Merge in Best_FP3 for the upcoming year
year_up = up.Year.iloc[0]
best_fp3_up = load_best_fp3(year_up)[['Driver','Best_FP3']]
up = up.merge(best_fp3_up, on='Driver', how='left')

# 4) Compute session‐relative gaps
for df in (train, up):
    for sess in ['FP1','FP2','FP3']:
        col = f'Best_{sess}'
        gap = f'Gap_{sess}'
        sb = df.groupby('Year')[col].transform('min')
        df[gap] = df[col] - sb

# 5) Compute the improvement deltas (on gaps)
train['Delta_FP2_FP1'] = train['Gap_FP2'] - train['Gap_FP1']
train['Delta_FP3_FP2'] = train['Gap_FP3'] - train['Gap_FP2']
up   ['Delta_FP2_FP1'] = up   ['Gap_FP2'] - up   ['Gap_FP1']
up   ['Delta_FP3_FP2'] = up   ['Gap_FP3'] - up   ['Gap_FP2']

# 6) Drop all raw Best_FP× columns and any old Q‐delta
to_drop = [f'Best_{s}' for s in ['FP1','FP2','FP3']] + ['Delta_Q_FP3']
train.drop(columns=to_drop, inplace=True, errors='ignore')
up   .drop(columns=to_drop, inplace=True, errors='ignore')

# 7) Rescale Elo & TeamElo, normalize RecencyWeight → [0,1]
for df in (train, up):
    df['Elo']         = df['Elo']     / 1000
    df['TeamElo']     = df['TeamElo'] / 1000
    rw = df['RecencyWeight']
    df['RecencyWeight'] = (rw - rw.min()) / (rw.max() - rw.min())

# 8) Save the final feature tables
train.to_csv('features_train_final.csv',    index=False)
up   .to_csv('features_upcoming_final.csv', index=False)

print("✔ Wrote features_train_final.csv and features_upcoming_final.csv")
