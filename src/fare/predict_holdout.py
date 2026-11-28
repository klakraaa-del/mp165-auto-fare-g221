"""Generate the required two-column holdout prediction file.

Run from repository root after DVC has materialized the raw CSVs:
    python -m src.fare.predict_holdout
"""
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.fare.cleaning import clean_features, load_params


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    params = load_params()
    raw_path = ROOT / params["data"]["holdout_path"]
    raw = pd.read_csv(raw_path)

    if "record_id" not in raw.columns:
        raise ValueError("holdout_features.csv must contain record_id")

    features = params["features"]
    model_columns = features["numeric"] + features["categorical"]
    cleaned = clean_features(raw, params)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    model_name = params["mlflow"]["model_name"]
    model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")

    predictions = model.predict(cleaned[model_columns])
    output = pd.DataFrame({
        "record_id": raw["record_id"],
        "prediction": predictions,
    })

    if len(output) != 250:
        raise ValueError(f"Expected 250 holdout rows, found {len(output)}")
    if list(output.columns) != ["record_id", "prediction"]:
        raise ValueError("Output schema must be record_id,prediction")

    destination = ROOT / "predictions" / "holdout_predictions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"Saved {len(output)} predictions to {destination}")
    print(output.head())


if __name__ == "__main__":
    main()
