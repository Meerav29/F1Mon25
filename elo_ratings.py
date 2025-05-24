# elo_ratings.py
import fastf1 as ff1
import pandas as pd

def fetch_all_results(years):
    records = []
    for year in years:
        for gp in ff1.get_event_schedule(year)['EventName']:
            sess = ff1.get_session(year, gp, 'R')
            sess.load()
            results = sess.results

            # Drop any rows where Position is NaN or non-numeric
            results = results[results['Position'].notna()]

            for _, row in results.iterrows():
                pos = row['Position']
                # ensure it’s a string of digits
                if isinstance(pos, str) and pos.isdigit():
                    finish = int(pos)
                elif isinstance(pos, (int,float)):
                    finish = int(pos)
                else:
                    # skip weird values like 'DNF'
                    continue

                records.append({
                    'Year': year,
                    'GP': gp,
                    'Driver': row['Abbreviation'],
                    'Team': row['TeamName'],
                    'FinishPos': finish
                })
    return pd.DataFrame(records)


def update_elo(ratings, group, k=20):
    """
    ratings: dict driver→current Elo
    group: DataFrame of this race’s [Driver, FinishPos]
    """
    drivers = group['Driver'].tolist()
    # for each pair (i,j), assign score=1 if i finished ahead of j, else 0
    for i, di in group.iterrows():
        for j, dj in group.iterrows():
            if di['Driver']==dj['Driver']: 
                continue
            Ri, Rj = ratings[di['Driver']], ratings[dj['Driver']]
            # expected score
            Ei = 1 / (1 + 10**((Rj - Ri)/400))
            Si = 1.0 if di['FinishPos'] < dj['FinishPos'] else 0.0
            ratings[di['Driver']] += k * (Si - Ei)
    return ratings

def compute_driver_elo(years, base_rating=1500, k=20):
    results = fetch_all_results(years)
    drivers = results['Driver'].unique()
    # init
    ratings = {d: base_rating for d in drivers}

    # iterate races in chronological order
    for (year, gp), grp in results.groupby(['Year','GP'], sort=True):
        ratings = update_elo(ratings, grp, k=k)

    # produce DataFrame of end-of-season ratings
    return pd.DataFrame([
        {'Year': year, 'Driver': d, 'Elo': ratings[d]}
        for year in years for d in drivers
    ])

