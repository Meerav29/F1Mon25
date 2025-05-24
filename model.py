# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import kendalltau

# 1) Load
train = pd.read_csv('features_train.csv')
up    = pd.read_csv('features_upcoming.csv')

# 2) Define columns to drop (identifiers, target, and any non-numeric)
drop_cols = ['Year','Driver','Best_Q','Team', 'RecencyWeight']
X_train = train.drop(drop_cols, axis=1)
y_train = train['Best_Q']

# 3) Weights (you can keep your 10× for 2024, or switch to use RecencyWeight)
# weights = np.where(train.Year==2024, 10, 1)
weights = train['RecencyWeight']

# 4) Fit the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train, sample_weight=weights)

# 5) Predict the upcoming grid
X_up = up.drop(['Year','Driver','Team'], axis=1)
up['PredTime'] = model.predict(X_up)
up['PredRank'] = up['PredTime'].rank(method='first')

# 6) Print your Monaco 2025 prediction
print(
    up
    .sort_values('PredRank')
    [['Driver','PredRank','PredTime']]
    .to_string(index=False)
)
