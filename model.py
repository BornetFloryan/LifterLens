import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

CSV_PATH = "database/openpowerlifting-2024-01-06-4c732975.csv"


def load_and_clean():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    df = df[df["Event"] == "SBD"]
    df = df[df["Equipment"] == "Raw"]

    df = df[["Sex", "Age", "BodyweightKg",
             "Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg"]]

    df = df[~df["Sex"].str.upper().eq("MX")]
    df = df.dropna()

    return df


def train_models():
    df = load_and_clean()

    X = df[["Sex", "Age", "BodyweightKg"]]
    targets = ["Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), ["Sex"])
        ],
        remainder="passthrough"
    )

    models = {}

    for t in targets:
        y = df[t]

        model = Pipeline(steps=[
            ("preprocess", preprocess),
            ("regressor", XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                eval_metric="mae"
            ))
        ])

        print(f"Entraînement du modèle pour : {t}")
        model.fit(X, y)
        models[t] = model

    with open("models.pkl", "wb") as f:
        pickle.dump(models, f)

    print("Modèles entraînés et sauvegardés dans models.pkl")


def load_models():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)


def predict_lifts(sex: str, age: int, bodyweight: float):
    models = load_models()

    sample = pd.DataFrame({
        "Sex": [sex],
        "Age": [age],
        "BodyweightKg": [bodyweight]
    })

    return {
        lift: float(models[lift].predict(sample)[0])
        for lift in models
    }


if __name__ == "__main__":
    train_models()