from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def build_xgboost_ic_feature_frame(
    panel: pd.DataFrame,
    factor_ic_frame: pd.DataFrame,
    factor_columns: list[str],
    ic_lag_months: list[int],
    ic_rolling_means: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    market_columns = ["spy_return_1m", "spy_drawdown", "vix_close", "vix_change_1m"]
    market_frame = panel[["Date", *market_columns]].drop_duplicates(subset=["Date"]).sort_values("Date").copy()

    feature_frame = factor_ic_frame.merge(market_frame, on="Date", how="left").sort_values("Date").reset_index(drop=True)

    feature_columns = ["spy_return_1m", "spy_drawdown", "vix_close", "vix_change_1m"]
    for factor_column in factor_columns:
        ic_column = f"{factor_column}_ic"
        for lag in ic_lag_months:
            column_name = f"{factor_column}_ic_lag{lag}"
            feature_frame[column_name] = feature_frame[ic_column].shift(lag)
            feature_columns.append(column_name)
        for window in ic_rolling_means:
            column_name = f"{factor_column}_ic_rollmean_{window}"
            feature_frame[column_name] = feature_frame[ic_column].shift(1).rolling(window, min_periods=1).mean()
            feature_columns.append(column_name)

    return feature_frame, feature_columns


def predict_factor_ic_with_xgboost(
    feature_frame: pd.DataFrame,
    factor_columns: list[str],
    feature_columns: list[str],
    training_window_months: int,
    min_training_rows: int,
    xgb_params: dict,
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
            train_slice = train_slice.dropna(subset=[target_column])
            if len(train_slice) < min_training_rows:
                predicted_values.append(np.nan)
                continue

            X_train = train_slice[feature_columns]
            y_train = train_slice[target_column]
            X_pred = feature_frame.iloc[[idx]][feature_columns]

            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            predicted_values.append(float(model.predict(X_pred)[0]))

        predictions[prediction_column] = predicted_values

    return predictions


def build_predicted_ic_weights(
    prediction_frame: pd.DataFrame,
    factor_columns: list[str],
    negative_prediction_weight: float,
    fallback: str,
    fallback_weights: dict[str, float] | None = None,
    shrinkage_ic_weight: float | None = None,
    shrinkage_baseline_weight: float | None = None,
) -> pd.DataFrame:
    weights_frame = prediction_frame[["Date"]].copy()
    weight_columns: list[str] = []

    for factor_column in factor_columns:
        prediction_column = f"{factor_column}_predicted_ic"
        weight_column = f"{factor_column}_weight"
        weights_frame[weight_column] = prediction_frame[prediction_column].clip(lower=negative_prediction_weight)
        weight_columns.append(weight_column)

    raw_weight_sum = weights_frame[weight_columns].sum(axis=1, min_count=len(weight_columns))
    valid_weight_rows = raw_weight_sum > 0
    for weight_column in weight_columns:
        weights_frame[weight_column] = np.where(
            valid_weight_rows,
            weights_frame[weight_column] / raw_weight_sum,
            np.nan,
        )

    if fallback in {"fixed_weight_baseline", "neutral_prior"}:
        if fallback_weights is None:
            raise ValueError("fallback_weights is required when fallback is fixed_weight_baseline or neutral_prior")
        for factor_column in factor_columns:
            weight_column = f"{factor_column}_weight"
            weights_frame[weight_column] = weights_frame[weight_column].fillna(fallback_weights[factor_column])
    elif fallback == "equal_factor_weight":
        equal_weight = 1.0 / len(weight_columns)
        for weight_column in weight_columns:
            weights_frame[weight_column] = weights_frame[weight_column].fillna(equal_weight)
    else:
        raise ValueError(f"Unsupported fallback strategy: {fallback}")

    if shrinkage_ic_weight is not None or shrinkage_baseline_weight is not None:
        if fallback_weights is None:
            raise ValueError("fallback_weights is required when shrinkage is enabled")
        if shrinkage_ic_weight is None or shrinkage_baseline_weight is None:
            raise ValueError("Both shrinkage weights must be provided")
        for factor_column in factor_columns:
            weight_column = f"{factor_column}_weight"
            baseline_weight = fallback_weights[factor_column]
            weights_frame[weight_column] = (
                shrinkage_ic_weight * weights_frame[weight_column].fillna(baseline_weight)
                + shrinkage_baseline_weight * baseline_weight
            )

    return weights_frame


def apply_predicted_ic_score(
    panel: pd.DataFrame,
    weights_frame: pd.DataFrame,
    factor_columns: list[str],
) -> pd.DataFrame:
    merged = panel.merge(weights_frame, on="Date", how="left")
    score = 0.0
    for factor_column in factor_columns:
        score = score + merged[factor_column] * merged[f"{factor_column}_weight"]
    merged["xgboost_ic_score"] = score
    return merged
