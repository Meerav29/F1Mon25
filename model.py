import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model(train_path="features_train_with_tyres.csv"):
    """Train a RandomForestRegressor on the given feature table."""
    train = pd.read_csv(train_path)
    drop_cols = ['Year', 'Driver', 'Team', 'Best_Q']
    X_train = train.drop(columns=drop_cols)
    y_train = train['Best_Q']
    weights = train['RecencyWeight']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def predict_upcoming(model, upcoming_path="features_upcoming_with_tyres.csv"):
    """Predict upcoming qualifying times using the provided model."""
    up = pd.read_csv(upcoming_path)
    X_up = up.drop(columns=['Year', 'Driver', 'Team'])
    up['PredTime'] = model.predict(X_up)
    up['PredRank'] = up['PredTime'].rank(method='first')
    return up[['Driver', 'PredRank', 'PredTime']].sort_values('PredRank')


def main(train_path="features_train_with_tyres.csv",
         upcoming_path="features_upcoming_with_tyres.csv"):
    model = train_model(train_path)
    results = predict_upcoming(model, upcoming_path)
    print(results.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train model and predict upcoming qualifying times"
    )
    parser.add_argument(
        "--train",
        default="features_train_with_tyres.csv",
        help="CSV with training features",
    )
    parser.add_argument(
        "--upcoming",
        default="features_upcoming_with_tyres.csv",
        help="CSV with upcoming race features",
    )
    args = parser.parse_args()

    main(args.train, args.upcoming)
