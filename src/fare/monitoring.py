
"""Stage 4 monitoring and v2 promotion for MP165.

Run from the repository root:
    python -m src.fare.monitoring
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from src.fare.cleaning import clean_features, clean_train, load_params
from src.fare.train import build_pipeline


ROOT = Path(__file__).resolve().parents[2]
FINDINGS_PATH = ROOT / "reports" / "findings.yaml"


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """PSI using reference-data quantile bins and a small zero-count correction."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))

    if len(edges) < 2:
        return 0.0

    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)

    ref_pct = np.maximum(ref_counts / ref_counts.sum(), 1e-6)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), 1e-6)

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def main() -> None:
    params = load_params()

    data = params["data"]
    features = params["features"]

    numeric = features["numeric"]
    categorical = features["categorical"]
    target = features["target"]

    model_name = params["mlflow"]["model_name"]

    train_raw = pd.read_csv(
        ROOT / data["train_path"],
        parse_dates=["date"],
    )

    later_raw = pd.read_csv(
        ROOT / data["later_path"],
        parse_dates=["date"],
    )

    train = clean_train(train_raw, params)
    later = clean_features(later_raw, params)

    # 1. PSI: compare training features with the later dataset.
    psi = {
        col: population_stability_index(
            train[col],
            later[col],
            bins=10,
        )
        for col in numeric
    }

    most_drifted = max(
        psi,
        key=lambda col: psi[col] if np.isfinite(psi[col]) else -1,
    )

    # Load registered v1 explicitly.
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    client = MlflowClient()

    v1 = mlflow.sklearn.load_model(
        f"models:/{model_name}/1"
    )

    # 2. v1 MAE by zone on the later dataset.
    later_x = later[numeric + categorical]
    later_y = later[target]

    later_pred = v1.predict(later_x)

    later_eval = later.assign(
        _absolute_error=np.abs(
            later_y.to_numpy() - later_pred
        )
    )

    mae_by_group = (
        later_eval.groupby("zone")["_absolute_error"]
        .mean()
        .to_dict()
    )

    worst_group = max(mae_by_group, key=mae_by_group.get)

    # 3. Chronological split: first 70% for training, last 30% for checking.
    later_sorted = later.sort_values("date").reset_index(drop=True)

    split_at = int(len(later_sorted) * 0.70)

    later_fit = later_sorted.iloc[:split_at]
    later_check = later_sorted.iloc[split_at:]

    combined = pd.concat(
        [train, later_fit],
        ignore_index=True,
    )

    estimator = clone(v1.named_steps["model"])

    v2 = build_pipeline(
        estimator,
        numeric,
        categorical,
    )

    v2.fit(
        combined[numeric + categorical],
        combined[target],
    )

    check_x = later_check[numeric + categorical]
    check_y = later_check[target]

    v1_check_pred = v1.predict(check_x)
    v2_check_pred = v2.predict(check_x)

    mae_v1 = float(
        mean_absolute_error(check_y, v1_check_pred)
    )

    mae_v2 = float(
        mean_absolute_error(check_y, v2_check_pred)
    )

    # 4. Promote v2 only if its check MAE is at least 15% lower.
    improvement = (
        (mae_v1 - mae_v2) / mae_v1
        if mae_v1 > 0
        else 0.0
    )

    promote = improvement >= 0.15

    with mlflow.start_run(
        run_name="v2_chronological_monitoring"
    ) as run:

        mlflow.log_param("model_name", "v2")
        mlflow.log_param("training_rows", len(combined))
        mlflow.log_param("later_training_fraction", 0.70)
        mlflow.log_param("check_rows", len(later_check))

        mlflow.log_metric("v1_check_mae", mae_v1)
        mlflow.log_metric("v2_check_mae", mae_v2)
        mlflow.log_metric(
            "mae_improvement_fraction",
            improvement,
        )

        model_info = mlflow.sklearn.log_model(
            v2,
            name="model_v2",
            skops_trusted_types=["numpy.dtype"],
        )

    registered = mlflow.register_model(
        model_info.model_uri,
        model_name,
    )

    if promote:
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=registered.version,
        )
        decision = "promoted_v2"
    else:
        decision = "kept_v1_champion"

    # 5. Write measured results to findings.yaml.
    with FINDINGS_PATH.open(encoding="utf-8") as stream:
        findings = yaml.safe_load(stream)

    experiment = client.get_experiment_by_name(
        params["mlflow"]["experiment_name"]
    )

    if experiment is None:
        raise RuntimeError("MLflow experiment was not found.")

    best_runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["metrics.validation_mae ASC"],
        max_results=1,
    )

    if not best_runs:
        raise RuntimeError("No completed MLflow training runs found.")

    best_run = best_runs[0]

    findings["tracking"] = {
        "experiment_name": params["mlflow"]["experiment_name"],
        "best_run_id": best_run.info.run_id,
        "registered_model_name": model_name,
        "champion_version": (
            registered.version if promote else 1
        ),
    }

    findings["monitoring"] = {
        "psi": {
            key: float(value) if np.isfinite(value) else None
            for key, value in psi.items()
        },
        "most_drifted_feature": most_drifted,
        "mae_by_group": {
            str(key): float(value)
            for key, value in mae_by_group.items()
        },
        "worst_served_group": str(worst_group),
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "decision": decision,
    }

    with FINDINGS_PATH.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            findings,
            stream,
            sort_keys=False,
            allow_unicode=True,
        )

    print("PSI:", psi)
    print("Most drifted feature:", most_drifted)
    print("MAE by zone:", mae_by_group)
    print("Worst-served zone:", worst_group)

    print(f"v1 check MAE: {mae_v1:.4f}")
    print(f"v2 check MAE: {mae_v2:.4f}")

    print(f"Improvement: {improvement:.1%}")
    print("Decision:", decision)
    print("Registered v2 version:", registered.version)


if __name__ == "__main__":
    main()
