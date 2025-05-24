# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import kendalltau

df = pd.read_csv('features_wide.csv')

train = df[df.Year == 2023].reset_index(drop=True)
test  = df[df.Year == 2024].reset_index(drop=True)

drop_cols = ['Year','Driver','Best_Q','RecencyWeight','Team']
features = [c for c in df.columns if c not in drop_cols]

X_train, y_train = train[features], train.Best_Q
X_test,  y_test  = test[features],  test.Best_Q


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE (secs): {mae:.3f}")

test = test.copy()
test['PredTime'] = y_pred

test['TrueRank'] = test.Best_Q.rank(method='first')
test['PredRank'] = test.PredTime.rank(method='first')

tau, p = kendalltau(test.TrueRank, test.PredRank)
print(f"Kendall’s Tau: {tau:.3f} (p={p:.3f})")

print(test[['Driver','Best_Q','PredTime','TrueRank','PredRank']].sort_values('PredRank').to_string(index=False))
