
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .cleaning import clean_train, load_params

def get_data_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_pipeline(model, numeric_features, categorical_features):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def main():

    params = load_params()

    train_path = params["data"]["train_path"]

    # Hash the original CSV before cleaning
    data_hash = get_data_hash(train_path)

    # Load and clean training data
    df = pd.read_csv(train_path, parse_dates=["date"])
    df = clean_train(df, params)

    numeric_features = params["features"]["numeric"]
    categorical_features = params["features"]["categorical"]
    target = params["features"]["target"]

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"]
    )

    models = {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=params["model"]["random_state"],
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=params["model"]["random_state"]
        )
    }

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    results = []

    for model_name, model in models.items():

        pipeline = build_pipeline(
            model,
            numeric_features,
            categorical_features
        )

        with mlflow.start_run(run_name=model_name) as run:

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_valid)

            mae = mean_absolute_error(y_valid, predictions)

            # Log experiment parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param(
                "random_state",
                params["model"]["random_state"]
            )
            mlflow.log_param(
                "test_size",
                params["model"]["test_size"]
            )
            mlflow.log_param("data_sha256", data_hash)

            for key, value in model.get_params().items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    mlflow.log_param(f"model_{key}", str(value))

            # Log validation metric
            mlflow.log_metric("validation_mae", mae)

            signature = infer_signature(X_valid, predictions)

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=X_valid.head(5),
                serialization_format="cloudpickle"
            )
            results.append({
                "model_name": model_name,
                "mae": mae,
                "model_uri": model_info.model_uri,
                "run_id": run.info.run_id
            })

            print(f"{model_name} MAE: {mae:.4f}")
            print(f"Run ID: {run.info.run_id}")
            print("-" * 40)

    # Find the model with the lowest MAE
    best = min(results, key=lambda result: result["mae"])

    print("\nBest model:", best["model_name"])
    print("Best MAE:", best["mae"])

    # Register the best model
    model_name = params["mlflow"]["model_name"]

    registered = mlflow.register_model(
        model_uri=best["model_uri"],
        name=model_name
    )

    # Assign champion alias
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=registered.version
    )

    print("\nModel registered successfully")
    print("Registered model:", model_name)
    print("Version:", registered.version)
    print("Alias: champion")


if __name__ == "__main__":
    main()
