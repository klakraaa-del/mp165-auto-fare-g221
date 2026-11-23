from sklearn.linear_model import LinearRegression
from src.fare.train import build_pipeline
import pandas as pd

from src.fare.cleaning import clean_booking, fix_units, correct_typos


def test_clean_booking_normalizes_categories():
    values = pd.Series([" App ", "str.", "STA", "Street"])

    result = clean_booking(
        values,
        {"str": "street", "sta": "stand"}
    )

    assert result.tolist() == ["app", "street", "stand", "street"]


def test_fix_units_converts_station_area_wait_time():
    df = pd.DataFrame({
        "zone": ["Market", "Station Area", "Outskirts"],
        "wait_min": [15.0, 900.0, 20.0]
    })

    params = {
        "cleaning": {
            "unit_group_column": "zone",
            "unit_group_value": "Station Area",
            "unit_column": "wait_min",
            "unit_factor": 60
        }
    }

    result = fix_units(df.copy(), params)

    assert result.loc[1, "wait_min"] == 15.0
    assert result.loc[0, "wait_min"] == 15.0
    assert result.loc[2, "wait_min"] == 20.0


def test_correct_typos_divides_extreme_fares():
    df = pd.DataFrame({
        "fare_rs": [100, 150, 200, 2500]
    })

    result = correct_typos(
        df.copy(),
        "fare_rs",
        multiplier=3,
        correction_factor=10
    )

    assert result.loc[3, "fare_rs"] == 250.0
    assert result.loc[0, "fare_rs"] == 100


def test_model_pipeline_trains_and_predicts():
    numeric_features = ["distance_km", "wait_min"]
    categorical_features = ["booking", "zone"]

    df = pd.DataFrame({
        "distance_km": [2, 4, 6, 8, 10, 12],
        "wait_min": [5, 10, 15, 20, 25, 30],
        "booking": ["app", "street", "stand", "app", "street", "stand"],
        "zone": ["Market", "Market", "Station Area", "Outskirts", "Market", "Outskirts"],
        "fare_rs": [50, 80, 110, 140, 170, 200]
    })

    X = df[numeric_features + categorical_features]
    y = df["fare_rs"]

    pipeline = build_pipeline(
        LinearRegression(),
        numeric_features,
        categorical_features
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert all(pd.notna(predictions))