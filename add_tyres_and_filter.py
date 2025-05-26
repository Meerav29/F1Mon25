import pandas as pd
from pathlib import Path


def tyre_stats(year):
    # Read in lap data for the year
    df = pd.read_parquet(Path("data/monaco") / f"monaco_{year}.parquet")
    df['LapSeconds'] = df.LapTime.dt.total_seconds()
    records = []
    for sess in ['FP1','FP2','FP3']:
        sub = df[df.Session == sess]
        grp = sub.groupby('Driver')
        stats = grp.agg(
            TotalLaps=('LapSeconds','size'),
            SoftLaps=('Compound', lambda c: (c=='SOFT').sum()),
            MeanSoftLap=('LapSeconds', lambda x: x[sub.loc[x.index,'Compound']=='SOFT'].mean())
        ).reset_index()
        stats['Year'] = year
        stats['Session'] = sess
        records.append(stats)
    return pd.concat(records, ignore_index=True)

# 1) Load the feature tables that include FP gaps & rescaled Elo/Recency
train = pd.read_csv('features_train_final.csv')
up    = pd.read_csv('features_upcoming_final.csv')

# 2) Compute tyre statistics for each historical year
years = sorted(train.Year.unique())
tyres = pd.concat([tyre_stats(y) for y in years], ignore_index=True)

# 3) Pivot tyre stats wide: one row per (Year,Driver)
tyre_wide = (
    tyres
    .pivot(index=['Year','Driver'], columns='Session',
           values=['TotalLaps','SoftLaps','MeanSoftLap'])
)
# Flatten the MultiIndex columns
tyre_wide.columns = [f"{stat}_{sess}" for stat, sess in tyre_wide.columns]
tyre_wide = tyre_wide.reset_index()

# 4) Merge the tyre features into train & upcoming
train = train.merge(tyre_wide, on=['Year','Driver'], how='left')
up    = up.merge(
    tyre_wide[tyre_wide.Year == up.Year.iloc[0]],
    on=['Year','Driver'], how='left'
)

# 5) Filter out drivers with too few laps (<8 total)
for df in (train, up):
    df['AllLaps'] = df['TotalLaps_FP1'] + df['TotalLaps_FP2'] + df['TotalLaps_FP3']
train = train[train.AllLaps >= 8].reset_index(drop=True)
up    = up[up.AllLaps       >= 8].reset_index(drop=True)

# 6) Save the updated feature sets
train.to_csv('features_train_with_tyres.csv', index=False)
up   .to_csv('features_upcoming_with_tyres.csv', index=False)

print("✔ Wrote features_train_with_tyres.csv and features_upcoming_with_tyres.csv")
