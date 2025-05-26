import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

train = pd.read_csv('features_train_with_fp3.csv')
drop = ['Year','Driver','Team','Best_Q','Delta_Q_FP3']
X = train.drop(drop,axis=1)
y = train.Best_Q
w = train.RecencyWeight

m = RandomForestRegressor(n_estimators=200,random_state=42)
m.fit(X, y, sample_weight=w)

plt.barh(X.columns, m.feature_importances_)
plt.title("Feature importances")
plt.tight_layout()
plt.show()
