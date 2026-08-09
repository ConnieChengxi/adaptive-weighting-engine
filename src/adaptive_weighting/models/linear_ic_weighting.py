from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler


LINEAR_MODEL_REGISTRY = {
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
}


def predict_factor_ic_with_linear_model(
    feature_frame: pd.DataFrame,
    factor_columns: list[str],
    feature_columns: list[str],
    training_window_months: int,
    min_training_rows: int,
    estimator_type: str,
    estimator_params: dict,
) -> pd.DataFrame:
    if estimator_type not in LINEAR_MODEL_REGISTRY:
        raise ValueError(f"Unsupported linear estimator type: {estimator_type}")

    estimator_cls = LINEAR_MODEL_REGISTRY[estimator_type]
    predictions = feature_frame[["Date"]].copy()

    for factor_column in factor_columns:
        target_column = f"{factor_column}_ic"
        prediction_column = f"{factor_column}_predicted_ic"
        predicted_values: list[float] = []

        for idx in range(len(feature_frame)):
            if idx < training_window_months:
                predicted_values.append(np.nan)
                continue

            train_slice = feature_frame.iloc[idx - training_window_months : idx].copy()
            train_slice = train_slice.dropna(subset=[target_column, *feature_columns])
            if len(train_slice) < min_training_rows:
                predicted_values.append(np.nan)
                continue

            X_train = train_slice[feature_columns]
            y_train = train_slice[target_column]
            X_pred = feature_frame.iloc[[idx]][feature_columns]
            if X_pred.isna().any(axis=None):
                predicted_values.append(np.nan)
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_pred_scaled = scaler.transform(X_pred)

            model = estimator_cls(**estimator_params)
            model.fit(X_train_scaled, y_train)
            predicted_values.append(float(model.predict(X_pred_scaled)[0]))

        predictions[prediction_column] = predicted_values

    return predictions
