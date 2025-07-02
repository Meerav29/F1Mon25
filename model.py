import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model():
    """Train a RandomForestRegressor on the tyre-enhanced feature table."""
    train = pd.read_csv('features_train_with_tyres.csv')
    drop_cols = ['Year', 'Driver', 'Team', 'Best_Q']
    X_train = train.drop(columns=drop_cols)
    y_train = train['Best_Q']
    weights = train['RecencyWeight']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def predict_upcoming(model):
    """Predict upcoming qualifying times using the provided model."""
    up = pd.read_csv('features_upcoming_with_tyres.csv')
    X_up = up.drop(columns=['Year', 'Driver', 'Team'])
    up['PredTime'] = model.predict(X_up)
    up['PredRank'] = up['PredTime'].rank(method='first')
    return up[['Driver', 'PredRank', 'PredTime']].sort_values('PredRank')


def main():
    model = train_model()
    results = predict_upcoming(model)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
