# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 1) Load the tyre‐enhanced, gap‐&‐delta feature tables
train = pd.read_csv('features_train_with_tyres.csv')
up    = pd.read_csv('features_upcoming_with_tyres.csv')

# 2) Split into X / y / weights
#    Drop the non-feature columns
drop_cols = ['Year', 'Driver', 'Team', 'Best_Q']
X_train   = train.drop(columns=drop_cols)
y_train   = train['Best_Q']
weights   = train['RecencyWeight']

# 3) Fit
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train, sample_weight=weights)

# 4) Predict
X_up = up.drop(columns=['Year','Driver','Team'])
up['PredTime'] = model.predict(X_up)
up['PredRank'] = up['PredTime'].rank(method='first')

# 5) Show
print(
    up
    .sort_values('PredRank')
    [['Driver','PredRank','PredTime']]
    .to_string(index=False)
)
