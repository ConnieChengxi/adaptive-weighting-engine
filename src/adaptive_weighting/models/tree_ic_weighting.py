from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def predict_factor_ic_with_random_forest(
    feature_frame: pd.DataFrame,
    factor_columns: list[str],
    feature_columns: list[str],
    training_window_months: int,
    min_training_rows: int,
    estimator_params: dict,
) -> pd.DataFrame:
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

            model = RandomForestRegressor(**estimator_params)
            model.fit(X_train, y_train)
            predicted_values.append(float(model.predict(X_pred)[0]))

        predictions[prediction_column] = predicted_values

    return predictions
