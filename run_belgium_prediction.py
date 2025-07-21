from model import train_model, predict_upcoming


def main():
    model = train_model()
    results = predict_upcoming(model, "features_belgium_2025_with_tyres.csv")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

